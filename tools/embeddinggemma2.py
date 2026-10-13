#!/usr/bin/env python3
"""Optional local EmbeddingGemma 2 worker; the CBM database is always read-only."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
import time

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
PROFILE = f"embeddinggemma-2:{REVISION}:768:fp32:code-retrieval-v1"
QUERY_PROMPT = "task: code retrieval | query: "
DIMENSION = 768
MAX_NODES = 200_000
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_DOCUMENT_BYTES = 256 * 1024 * 1024
MAX_SOURCE_LINES = 2_000_000


def runtime_versions():
    values = {}
    for package in ("torch", "sentence-transformers", "transformers"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "missing"
    return values


def opened_path(descriptor):
    """Resolve the opened handle before reading, so a path swap cannot escape the root."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        import msvcrt
        function = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
        function.argtypes = (wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD)
        function.restype = wintypes.DWORD
        buffer = ctypes.create_unicode_buffer(32768)
        length = function(msvcrt.get_osfhandle(descriptor), buffer, len(buffer), 0)
        if not 0 < length < len(buffer):
            raise ctypes.WinError(ctypes.get_last_error())
        raw = buffer.value
        if raw.startswith("\\\\?\\UNC\\"):
            raw = "\\\\" + raw[8:]
        elif raw.startswith("\\\\?\\"):
            raw = raw[4:]
    elif sys.platform == "darwin":
        import fcntl
        # Darwin sys/fcntl.h defines F_GETPATH=50, returning the path for this fd.
        raw = fcntl.fcntl(descriptor, 50, b"\0" * 1024).split(b"\0", 1)[0].decode()
    elif sys.platform.startswith("linux"):
        raw = os.readlink(f"/proc/self/fd/{descriptor}")
    else:
        raise OSError("source-handle verification is unsupported on this platform")
    return Path(raw)


def read_source(path, root):
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as file:
        info = os.fstat(file.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
            raise ValueError("source is not a bounded regular file")
        if not opened_path(file.fileno()).is_relative_to(root):
            raise ValueError("source handle leaves the repository")
        data = file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("source file grew past the read limit")
        return data


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def snapshot(db_path, project):
    """Bind node ids and source text to one native, content-verified generation."""
    with closing(readonly(db_path)) as db, db:
        db.execute("BEGIN")
        metadata = dict(db.execute(
            "SELECT k,v FROM store_meta WHERE k IN ('db_uid','mutation_gen')"))
        uid, gen = metadata.get("db_uid", ""), metadata.get("mutation_gen", "")
        if len(uid) != 16 or any(c not in "0123456789abcdef" for c in uid) or (
                not gen.isascii() or not gen.isdecimal() or str(int(gen)) != gen):
            raise ValueError("CBM generation is missing or malformed; reindex with this build")
        generation = f"u{uid}g{gen}"
        row = db.execute("SELECT root_path FROM projects WHERE name=?", (project,)).fetchone()
        if not row:
            raise ValueError("project is not indexed")
        root = Path(row[0]).resolve(strict=True)
        rows = list(db.execute(
            "SELECT id,qualified_name,file_path,start_line,end_line FROM nodes "
            "WHERE project=? AND label IN ('Function','Method','Class') AND start_line>0 "
            "ORDER BY id LIMIT ?", (project, MAX_NODES + 1)))
        if len(rows) > MAX_NODES:
            raise ValueError("EmbeddingGemma 2 corpus exceeds the 200000-node limit")
        expected = dict(db.execute("SELECT rel_path,sha256 FROM file_hashes WHERE project=?",
                                   (project,)))
    file_lines, file_hashes, documents = {}, {}, []
    excluded_virtual = 0
    source_bytes, document_bytes, source_lines = 0, 0, 0
    for node_id, qn, relative, start, end in rows:
        if relative not in expected and relative.startswith('<') and relative.endswith('>'):
            excluded_virtual += 1
            continue
        if relative not in file_lines:
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("source path leaves the repository")
            data = read_source(path, root)
            source_bytes += len(data)
            if source_bytes > MAX_SOURCE_BYTES:
                raise ValueError("source corpus exceeds the 256 MiB limit")
            digest = hashlib.sha256(data).hexdigest()
            if expected.get(relative) != digest:
                raise ValueError(f"CBM source is stale or unverified: {relative}; reindex CBM")
            file_hashes[relative] = digest
            file_lines[relative] = data.decode("utf-8").splitlines(keepends=True)
            source_lines += len(file_lines[relative])
            if source_lines > MAX_SOURCE_LINES:
                raise ValueError("source corpus exceeds the 2000000-line limit")
        lines = file_lines[relative]
        if not 1 <= start <= end <= len(lines) + 1 or start > len(lines):
            raise ValueError(f"invalid native source span: {qn}")
        text = f"title: {relative} | text: {''.join(lines[start - 1:end])}"
        document_bytes += len(text.encode())
        if document_bytes > MAX_DOCUMENT_BYTES:
            raise ValueError('document corpus exceeds the 256 MiB limit')
        documents.append((node_id, hashlib.sha256(text.encode()).hexdigest(), text))
    identity = [str(Path(db_path).resolve()), project, generation, str(root), rows,
                sorted(file_hashes.items())]
    fingerprint = hashlib.sha256(json.dumps(identity, ensure_ascii=False,
                                            separators=(",", ":")).encode()).hexdigest()
    return generation, fingerprint, documents, excluded_virtual


def index_path(db_path):
    return Path(str(Path(db_path).resolve()) + ".embeddinggemma-2.db")


def verify_vector(blob):
    import numpy as np
    if len(blob) != DIMENSION * 4:
        raise ValueError("invalid embedding byte length")
    vector = np.frombuffer(blob, dtype="<f4")
    if not np.isfinite(vector).all() or not math.isclose(
            float(np.linalg.norm(vector)), 1.0, abs_tol=2e-4):
        raise ValueError("embedding is nonfinite or not L2-normalized")
    return vector


class Runtime:
    def __init__(self, allow_download=False):
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.device = os.environ.get("CBM_GEMMA2_DEVICE", "cpu")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("CBM_GEMMA2_DEVICE must be cpu or cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; CPU fallback is disabled")
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = SentenceTransformer(
            MODEL, revision=REVISION, device=self.device,
            local_files_only=not allow_download,
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": torch.float32})
        self.model.max_seq_length = 4096
        self.model.default_prompt_name = None
        self.model.eval()
        config = self.model[0].auto_model.config
        if self.model.get_embedding_dimension() != DIMENSION or (
                self.model.prompts.get("CodeRetrieval") != QUERY_PROMPT or
                getattr(config, "vision_config", None) is not None or
                getattr(config, "audio_config", None) is not None or
                any(p.device.type != self.device or (
                    p.is_floating_point() and p.dtype != torch.float32)
                    for p in self.model.parameters())):
            raise RuntimeError("model profile, text-only mode, precision, or device mismatch")
        parameters = sum(p.numel() for p in self.model.parameters())
        if abs(parameters - 270_000_000) > 2_000_000:
            raise RuntimeError("the expected text branch did not load")
        self.parameters = parameters
        self.truncated = 0

    def encode(self, texts, query=False):
        prefix = QUERY_PROMPT if query else ""
        self.truncated += sum(len(self.model.tokenizer(
            prefix + text, truncation=False)["input_ids"]) > 4096 for text in texts)
        vector = self.model.encode(texts, prompt_name="CodeRetrieval" if query else None,
                                   batch_size=1, normalize_embeddings=True,
                                   convert_to_tensor=True, show_progress_bar=False)
        if vector.device.type != self.device:
            raise RuntimeError("inference tensor left the selected device")
        array = vector.detach().to(device="cpu", dtype=self.torch.float32).numpy()
        blobs = [row.astype("<f4").tobytes() for row in array]
        for blob in blobs:
            verify_vector(blob)
        return blobs


def read_cache(path):
    if not path.exists():
        return None, []
    with closing(readonly(path)) as db, db:
        row = db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
        if not row:
            raise ValueError("EmbeddingGemma 2 index has no manifest")
        manifest = json.loads(row[0])
        rows = list(db.execute(
            "SELECT node_id,document_sha,vector,truncated FROM embeddings ORDER BY node_id LIMIT ?",
            (MAX_NODES + 1,)))
        if not isinstance(manifest, dict) or len(rows) > MAX_NODES:
            raise ValueError("EmbeddingGemma 2 cache is malformed or exceeds the corpus limit")
    return manifest, rows


def build_index(db_path, project, allow_download=False, runtime_factory=Runtime, _state=None):
    # One publisher across MCP sessions; SQLite releases this lock on process death.
    with closing(sqlite3.connect(str(index_path(db_path)) + ".lock", timeout=0)) as lock:
        try:
            lock.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as error:
            if error.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
            raise ValueError("EmbeddingGemma 2 index update is busy; retry after it finishes") from error
        return _build_index(db_path, project, allow_download, runtime_factory, _state)


def _build_index(db_path, project, allow_download, runtime_factory, state):
    generation, fingerprint, documents, excluded_virtual = state or snapshot(db_path, project)
    destination = index_path(db_path)
    previous, old_rows = read_cache(destination)
    compatible = previous and previous.get("profile") == PROFILE and (
        previous.get("runtime_versions") == runtime_versions())
    reusable = {row[1]: (row[2], row[3]) for row in old_rows} if compatible else {}
    if compatible and (
            previous.get("generation") == generation and
            previous.get("fingerprint") == fingerprint and
            previous.get("count") == len(documents) and
            [(r[0], r[1]) for r in old_rows] == [(d[0], d[1]) for d in documents]):
        for _, digest, _ in documents:
            if digest not in reusable:
                raise ValueError("EmbeddingGemma 2 index is incomplete")
            verify_vector(reusable[digest][0])
        return {**previous, "embedded": 0, "reused": len(documents)}
    descriptor, pending = tempfile.mkstemp(prefix=destination.name + ".", suffix=".pending",
                                           dir=destination.parent)
    os.close(descriptor)
    runtime, embedded, reused, truncation_count = None, 0, 0, 0
    try:
        with closing(sqlite3.connect(pending)) as db, db:
            db.executescript(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
                "CREATE TABLE embeddings(node_id INTEGER PRIMARY KEY, document_sha TEXT NOT NULL,"
                "vector BLOB NOT NULL CHECK(length(vector)=3072),"
                "truncated INTEGER NOT NULL CHECK(truncated IN (0,1)));")
            for node_id, digest, text in documents:
                if digest in reusable:
                    blob, truncated = reusable[digest]
                    verify_vector(blob)
                    reused += 1
                else:
                    if runtime is None:
                        runtime = runtime_factory(allow_download)
                    before = runtime.truncated
                    blob = runtime.encode([text])[0]
                    truncated = int(runtime.truncated > before)
                    embedded += 1
                    reusable[digest] = (blob, truncated)
                truncation_count += truncated
                db.execute("INSERT INTO embeddings VALUES(?,?,?,?)",
                           (node_id, digest, blob, truncated))
            if snapshot(db_path, project)[:2] != (generation, fingerprint):
                raise ValueError("CBM/source changed while embedding; old index preserved")
            manifest = {"profile": PROFILE, "generation": generation, "fingerprint": fingerprint,
                        "count": len(documents), "excluded_virtual_nodes": excluded_virtual,
                        "dimension": DIMENSION, "text_only": True,
                        "precision": "float32", "normalized": True,
                        "maximum_input_tokens": 4096, "query_prompt": QUERY_PROMPT,
                        "document_format": "title: {relative_path} | text: {code}",
                        "truncated_documents": truncation_count,
                        "runtime_versions": runtime_versions(),
                        "device": getattr(runtime, "device", previous.get("device") if compatible else None),
                        "loaded_parameters": getattr(runtime, "parameters",
                                                     previous.get("loaded_parameters") if compatible else None)}
            db.execute("INSERT INTO metadata VALUES('manifest',?)", (json.dumps(manifest),))
        os.replace(pending, destination)
        return {**manifest, "embedded": embedded, "reused": reused}
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def search(request, runtime_factory=Runtime):
    import numpy as np
    required = ("db_path", "project", "generation", "limit", "queries")
    if not isinstance(request, dict) or any(key not in request for key in required):
        raise ValueError("invalid worker request")
    queries, limit = request["queries"], request["limit"]
    if (not isinstance(queries, list) or not 1 <= len(queries) <= 32 or
            any(not isinstance(q, str) or not q.strip() or len(q.encode()) > 4096
                for q in queries) or type(limit) is not int or not 1 <= limit <= 100500 or
            any(not isinstance(request[k], str) or not request[k]
                for k in ("db_path", "project", "generation"))):
        raise ValueError("query or limit is outside the worker contract")
    state = snapshot(request["db_path"], request["project"])
    generation, fingerprint, documents, _ = state
    if generation != request["generation"]:
        raise ValueError("native index changed; retry with its current generation")
    manifest, rows = read_cache(index_path(request["db_path"]))
    if not manifest or manifest.get("profile") != PROFILE or (
            manifest.get("runtime_versions") != runtime_versions()):
        raise ValueError("EmbeddingGemma 2 index is missing/stale/incompatible; run index explicitly")
    runtime = None

    def get_runtime(allow_download=False):
        nonlocal runtime
        if runtime is None:
            runtime = runtime_factory(False)
        return runtime

    update = None
    if (manifest.get("generation") != generation or manifest.get("fingerprint") != fingerprint or
            manifest.get("count") != len(documents)):
        update = build_index(request["db_path"], request["project"],
                             runtime_factory=get_runtime, _state=state)
        manifest, rows = read_cache(index_path(request["db_path"]))
        print(f"EmbeddingGemma 2: index refreshed; embedded={update['embedded']} "
              f"reused={update['reused']}", file=sys.stderr, flush=True)
    if not manifest or manifest.get("generation") != generation or (
            manifest.get("profile") != PROFILE or manifest.get("fingerprint") != fingerprint or
            manifest.get("runtime_versions") != runtime_versions() or
            manifest.get("count") != len(documents)):
        raise ValueError("EmbeddingGemma 2 index changed during refresh; retry")
    if [(r[0], r[1]) for r in rows] != [(d[0], d[1]) for d in documents]:
        raise ValueError("EmbeddingGemma 2 index does not match the native nodes")
    if not rows:
        return {"profile": PROFILE, "generation": generation, "results": []}
    vectors = np.vstack([verify_vector(row[2]) for row in rows])
    runtime = get_runtime()
    query_vectors = np.vstack([verify_vector(blob) for blob in runtime.encode(queries, query=True)])
    scores = np.clip((vectors @ query_vectors.T).min(axis=1), -1.0, 1.0)
    ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    order = np.lexsort((ids, -scores))[:limit]
    if snapshot(request["db_path"], request["project"])[:2] != (generation, fingerprint):
        raise ValueError("CBM/source changed during search; results discarded")
    result = {"profile": PROFILE, "generation": generation,
              "results": [{"node_id": int(ids[i]), "score": float(scores[i])} for i in order]}
    if update:
        result["index_update"] = {"embedded": update["embedded"], "reused": update["reused"]}
    return result


@contextmanager
def parent_process(owner_pid):
    if owner_pid <= 0:
        raise ValueError("invalid worker owner pid")
    if os.name != "nt":
        if os.getppid() != owner_pid:
            raise ValueError("worker owner is not its parent process")
        yield lambda: os.getppid() == owner_pid
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    # Windows venv launchers insert a parent; monitor CBM, which owns the Job Object.
    handle = kernel.OpenProcess(0x00100000, False, owner_pid)  # SYNCHRONIZE only
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield lambda: kernel.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT
    finally:
        kernel.CloseHandle(handle)


def publish(directory, name, value):
    data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("worker response exceeds 8 MiB")
    pending = directory / f"{name}.pending"
    pending.write_bytes(data)
    os.replace(pending, directory / f"{name}.json")


def serve(directory, owner_pid, runtime_factory=Runtime):
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("worker directory is missing")
    with parent_process(owner_pid) as parent_alive:
        runtime = runtime_factory(False)
        publish(directory, "ready", {"profile": PROFILE, "status": "ready", "pid": os.getpid()})
        print("EmbeddingGemma 2: model ready; retaining runtime for this server", file=sys.stderr,
              flush=True)
        last_id = 0
        while parent_alive():
            path = directory / "request.json"
            if not path.exists():
                time.sleep(0.01)
                continue
            identifier = 0
            try:
                with path.open("rb") as file:
                    payload = file.read(256 * 1024 + 1)
                path.unlink()
                if len(payload) > 256 * 1024:
                    raise ValueError("worker request exceeds 256 KiB")
                request = json.loads(payload)
                identifier = request.get("request_id") if isinstance(request, dict) else None
                if type(identifier) is not int or not last_id < identifier <= 2**63 - 1:
                    raise ValueError("invalid or repeated worker request id")
                last_id = identifier
                result = search(request, lambda _: runtime)
                result["request_id"] = identifier
                publish(directory, "response", result)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                print(f"EmbeddingGemma 2: {message}", file=sys.stderr, flush=True)
                publish(directory, "response", {"request_id": identifier, "error": message[:2048]})
                if isinstance(error, RuntimeError):
                    return 1  # A failed inference runtime is restarted on the next request.
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    indexing = sub.add_parser("index")
    indexing.add_argument("--db", required=True)
    indexing.add_argument("--project", required=True)
    indexing.add_argument("--allow-download", action="store_true")
    sub.add_parser("search")
    serving = sub.add_parser("serve")
    serving.add_argument("--directory", required=True)
    serving.add_argument("--owner-pid", required=True, type=int)
    args = parser.parse_args()
    try:
        if args.operation == "serve":
            return serve(args.directory, args.owner_pid)
        if args.operation == "index":
            result = build_index(args.db, args.project, args.allow_download)
        else:
            payload = sys.stdin.buffer.read(256 * 1024 + 1)
            if len(payload) > 256 * 1024:
                raise ValueError("worker request exceeds 256 KiB")
            result = search(json.loads(payload))
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except Exception as error:
        print(f"EmbeddingGemma 2: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
