# Optional local EmbeddingGemma 2 search

This fork prototype adds `embedding_model: "embeddinggemma-2"` to
`search_graph`. Omitting it, or choosing `"default"`, uses the existing
semantic search. BM25 still uses `query`. The model option requires
`semantic_query`; unknown models and backend failures return an error.

The C binary invokes an explicitly configured local worker through the existing
supervised subprocess API. It does not bundle a transformer runtime or weights.
The reference worker in `tools/embeddinggemma2.py` uses Python 3.12,
sentence-transformers and PyTorch. Its dependencies are optional.

## Setup

1. Build CBM normally and index the repository in full mode with a project name:

   ```sh
   codebase-memory-mcp cli --json index_repository --repo-path /path/to/repo --name example --mode full
   ```

2. Create a local Python environment. Install a compatible PyTorch/torchvision
   pair using the [official PyTorch instructions](https://pytorch.org/get-started/locally/),
   then install `tools/requirements-embeddinggemma2.txt`. Choose the CPU or CUDA
   wheel explicitly before installing the optional requirements. This fork was
   exercised with PyTorch 2.14.0+cu126 and torchvision 0.29.0+cu126 on a GTX 1070 Ti.
   A newer CUDA wheel that excludes Pascal is unsuitable for that GPU.

3. Set these variables in the CBM server's process environment:

   ```text
   CBM_GEMMA2_PYTHON=/absolute/path/to/venv/python
   CBM_GEMMA2_SCRIPT=/absolute/path/to/checkout/tools/embeddinggemma2.py
   CBM_GEMMA2_DEVICE=cuda
   ```

   `CBM_GEMMA2_DEVICE=cpu` selects CPU explicitly; CPU is the worker's default.
   Requested CUDA must be available. No CPU fallback occurs after a CUDA request.
   With `CBM_GEMMA2_PRELOAD` unset or `0`, the first explicit Gemma search
   starts one supervised worker and loads the model. Later searches in that MCP
   session reuse it, including searches of other projects. Default-model and BM25
   searches do not start a Gemma worker.

   Optional `CBM_GEMMA2_PRELOAD=1` loads cached weights during MCP `initialize`,
   before its response. This moves cold-start time to connection setup; initialize
   returns an error if preload fails. Other nonempty values are rejected.
   Set the variables before starting CBM's daemon; an already running daemon
   retains its launch environment. Use a dedicated `CBM_CACHE_DIR` for isolation.

4. Create the independent Gemma index from the published CBM database:

   ```sh
   /path/to/venv/python tools/embeddinggemma2.py index --db /path/to/example.db --project example --allow-download
   ```

   Only this explicit `--allow-download` permits the first Hugging Face download.
   Later indexing and every search use cached weights only. Set `HF_HOME` to a
   local directory if desired. The model's license applies separately to its
   downloaded weights.

5. Search using the existing MCP tool, or equivalent CLI flags:

   ```json
   {
     "project": "example",
     "semantic_query": ["Where are failed HTTP requests retried?"],
     "embedding_model": "embeddinggemma-2",
     "semantic_limit": 5,
     "format": "json"
   }
   ```

   ```sh
   codebase-memory-mcp cli --json search_graph --project example --semantic-query '["HTTPリクエストを再試行する処理"]' --embedding-model embeddinggemma-2
   ```

For Gemma, each array entry is a complete query phrase. Multiple entries retain
the existing min-cosine ranking contract, using the lowest similarity per node.
Ties are resolved by ascending native node ID. Existing filtering, pagination
and output formatting are applied by CBM after the backend returns candidates.
Results identify code locations at the verified search snapshot. An editing
agent must reopen the current file immediately before modifying it; another
process can change source after a successful search.

## Index and trust boundaries

The index is a separate `<native-db>.embeddinggemma-2.db` file. Native vectors,
token statistics, graph edges and structural tools are not rewritten.
It contains FP32, 768-dimensional, finite, L2-normalized vectors for native
Function, Method and Class nodes with verifiable repository source.
Synthetic paths such as `<python-builtins>` are excluded and counted in the
manifest. Other missing or unverified source files fail indexing.

The profile fixes:

- Model: `google/embeddinggemma-2`
- Revision: `914f7f89142e33e77833254d9c9b90c3cef7303b`
- Query prefix: `task: code retrieval | query: `
- Document: `title: {relative_path} | text: {code}`
- Maximum input length: 4096 tokens; truncation is counted
- FP32 inference, dimension 768, normalization, text only

The manifest binds the vectors to the native store generation, node identities,
verified source hashes and inference-library versions. Indexing reuses unchanged
document vectors, writes a temporary database, then publishes it atomically.
A failed rebuild preserves the previous database. Source edits, native reindexing,
profile changes, corrupt vectors and mismatched generations cause search errors.
After source changes, let CBM's native watcher publish the updated graph, or
run index_repository explicitly. The next explicit Gemma search automatically
refreshes an existing compatible Gemma index using the already loaded model.
Only changed document hashes are encoded; unchanged functions are reused even
when native node IDs move. Added/deleted nodes are rebound in the new index.
The worker log reports embedded and reused counts. Initial index creation and
profile/library upgrades remain explicit commands.

A SQLite publisher lock serializes updates across MCP sessions and explicit
index commands; another publisher receives a busy error and can retry.
A failed refresh preserves the prior index and returns an error. Stale results
are not returned while the native graph or Gemma index is updating.

C validates the worker's profile, generation, result count, node ownership,
finite scores, ordering and duplicate IDs. It uses an owner-private request
directory, argument-vector spawning without a shell, a 60-second deadline for
model startup and each request, and process-tree supervision. Failed-worker logs stay in that private directory;
the server log gives their path. The worker reads CBM databases in read-only mode
and refuses source paths outside the repository.

The reference worker is deliberately bounded to 200000 source nodes and
16 MiB per source file, 256 MiB each for aggregate source and document text,
and two million source lines. Opened source handles are verified beneath the root
before their contents are read. Requests and replies are published atomically
in a private directory and matched by monotonically increasing IDs. No listening
port is opened. Search verifies current source and index contents every time.
Before and after each MCP Gemma search, CBM's own discovery/semantic manifest
is compared with the complete published native manifest, using its recorded
index mode and existing ignore/config rules. This also detects added/deleted
files and configuration changes before the watcher polls. Inspection failures
refuse the search. This inspection is bounded to 200000 files and 256 MiB of
discovered source. The guard reads/hashes native inputs without reparsing code;
it adds corpus-dependent I/O to search latency.

The loaded model and its CUDA allocation remain resident until the session ends.
Normal teardown terminates the owned process tree; on parent death Windows Job
Object containment or the worker's parent monitor releases it. A worker crash or
timeout fails the current search; the next explicit Gemma search starts a fresh
worker. Invalid or stale input returns an error while keeping a usable model.
There is no hidden retry of a failed request or fallback to another model or device.

The first lazy search includes imports/model loading; warmed searches do not.
Preload has the same load cost during initialization. Separate MCP sessions own
separate model processes, and one-shot CLI sessions cannot retain their runtime
after exit. The earlier full-corpus benchmark uses different document units and
does not measure this adapter's latency.

## Verification

```sh
python tests/test_embeddinggemma2.py -v
scripts/test.sh --suites semantic,mcp
```

The Python contract tests use a deterministic fake encoder and require NumPy,
without downloading weights. Persistent-worker tests cover reuse, request IDs,
stale source, incremental refresh, publisher locking and inference failures.
C tests cover model selection, response validation and native manifest freshness.
Real CUDA and MCP integration evidence is recorded separately in the local work report; the synthetic fixture is an integration check, not a retrieval
quality benchmark.

Upstream contribution requires a prior design discussion, as documented in
[CONTRIBUTING.md](../CONTRIBUTING.md). Issue
[#1112](https://github.com/DeusData/codebase-memory-mcp/issues/1112)
is related context; its external-companion discussion does not establish approval
of this new MCP parameter or the reference worker.
