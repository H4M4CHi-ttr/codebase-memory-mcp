"""Worker contract tests; model downloads and real inference are not required."""
import hashlib
from contextlib import closing, contextmanager
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location(
    "gemma_worker", Path(__file__).resolve().parents[1] / "tools" / "embeddinggemma2.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class FakeRuntime:
    calls = 0
    truncated = 0

    def __init__(self, allow_download=False):
        pass

    def encode(self, texts, query=False):
        FakeRuntime.calls += len(texts)
        result = []
        for text in texts:
            array = np.zeros(768, dtype="<f4")
            array[0 if "alpha" in text else 1] = 1
            result.append(array.tobytes())
        return result


class WorkerContract(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).resolve().parents[1] / "build" / "worker-tests"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="gemma2-", dir=self.temp_root)
        self.root = Path(self.tmp.name)
        self.source = self.root / "test.py"
        self.source.write_text("def alpha():\n    return 1\n\ndef beta():\n    return 2\n",
                               encoding="utf-8")
        self.db_path = self.root / "native.db"
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.executescript(
                "CREATE TABLE store_meta(k TEXT PRIMARY KEY,v TEXT);"
                "INSERT INTO store_meta VALUES('db_uid','0123456789abcdef');"
                "INSERT INTO store_meta VALUES('mutation_gen','1');"
                "CREATE TABLE projects(name TEXT,root_path TEXT);"
                "CREATE TABLE file_hashes(project TEXT,rel_path TEXT,sha256 TEXT);"
                "CREATE TABLE nodes(id INTEGER,project TEXT,label TEXT,qualified_name TEXT,"
                "file_path TEXT,start_line INTEGER,end_line INTEGER);")
            db.execute("INSERT INTO projects VALUES('test',?)", (str(self.root),))
            db.execute("INSERT INTO file_hashes VALUES('test','test.py',?)",
                       (hashlib.sha256(self.source.read_bytes()).hexdigest(),))
            for identifier, name, start, end in [(1, "alpha", 1, 2), (2, "beta", 4, 5)]:
                db.execute("INSERT INTO nodes VALUES(?,'test','Function',?,'test.py',?,?)",
                           (identifier, name, start, end))
        FakeRuntime.calls = 0

    def tearDown(self):
        assert self.root.resolve().parent == self.temp_root.resolve()
        self.tmp.cleanup()

    def request(self, queries=None, limit=2):
        return {"db_path": str(self.db_path), "project": "test",
                "generation": worker.snapshot(self.db_path, "test")[0],
                "queries": queries or ["alpha"], "limit": limit}

    def build(self):
        return worker.build_index(self.db_path, "test", runtime_factory=FakeRuntime)

    @contextmanager
    def running_worker(self, runtime_factory=FakeRuntime):
        directory = self.root / "ipc"
        directory.mkdir()
        alive = threading.Event()
        alive.set()
        errors = []

        @contextmanager
        def parent(_):
            yield alive.is_set

        def run():
            try:
                worker.serve(directory, 123, runtime_factory)
            except Exception as error:
                errors.append(error)

        with patch.object(worker, "parent_process", parent):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.wait_file(directory / "ready.json")
                yield directory
            finally:
                alive.clear()
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
        if errors:
            raise errors[0]

    def wait_file(self, path):
        deadline = time.monotonic() + 5
        while not path.exists():
            if time.monotonic() >= deadline:
                self.fail(f"worker did not publish {path.name}")
            time.sleep(0.01)
        return json.loads(path.read_bytes())

    def exchange(self, directory, request, identifier):
        worker.publish(directory, "request", {**request, "request_id": identifier})
        response = self.wait_file(directory / "response.json")
        (directory / "response.json").unlink()
        self.assertEqual(response["request_id"], identifier)
        return response

    def test_persistent_worker_loads_once_and_survives_invalid_request(self):
        self.build()
        loads = []

        def factory(allow_download):
            loads.append(allow_download)
            return FakeRuntime(allow_download)

        with self.running_worker(factory) as directory:
            ready = self.wait_file(directory / "ready.json")
            self.assertEqual((ready["profile"], ready["status"]), (worker.PROFILE, "ready"))
            first = self.exchange(directory, self.request(), 1)
            self.assertEqual(first["results"][0]["node_id"], 1)
            invalid = self.exchange(directory, {**self.request(), "queries": []}, 2)
            self.assertIn("contract", invalid["error"])
            repeated = self.exchange(directory, self.request(), 2)
            self.assertIn("repeated", repeated["error"])
            next_result = self.exchange(directory, self.request(["beta"]), 3)
            self.assertEqual(next_result["results"][0]["node_id"], 2)
            self.assertEqual(loads, [False])

    def test_persistent_worker_detects_source_edits_without_reloading(self):
        self.build()
        with self.running_worker() as directory:
            request = self.request()
            self.exchange(directory, request, 1)
            before = self.source.read_bytes()
            self.source.write_bytes(before + b"# changed\n")
            failed = self.exchange(directory, request, 2)
            self.assertIn("source is stale", failed["error"])
            self.source.write_bytes(before)
            self.assertIn("results", self.exchange(directory, request, 3))

    def test_persistent_worker_exits_on_runtime_failure(self):
        self.build()

        class FailingRuntime(FakeRuntime):
            def encode(self, texts, query=False):
                raise RuntimeError("simulated device failure")

        with self.running_worker(FailingRuntime) as directory:
            failed = self.exchange(directory, self.request(), 1)
            self.assertIn("simulated device failure", failed["error"])

    def test_build_search_and_resume(self):
        self.assertEqual(self.build()["embedded"], 2)
        self.assertEqual(self.build()["embedded"], 0)
        self.assertEqual(FakeRuntime.calls, 2)
        answer = worker.search(self.request(), FakeRuntime)
        self.assertEqual([r["node_id"] for r in answer["results"]], [1, 2])
        self.assertEqual(answer["profile"], worker.PROFILE)

    def test_stale_source_and_generation_fail_closed(self):
        self.build()
        request = self.request()
        self.source.write_text("def alpha():\n    return 9\n\ndef beta():\n    return 2\n")
        with self.assertRaisesRegex(ValueError, "source is stale"):
            worker.search(request, FakeRuntime)
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("UPDATE file_hashes SET sha256=?",
                       (hashlib.sha256(self.source.read_bytes()).hexdigest(),))
            db.execute("UPDATE store_meta SET v='2' WHERE k='mutation_gen'")
        with self.assertRaisesRegex(ValueError, "native index changed"):
            worker.search(request, FakeRuntime)
        answer = worker.search(self.request(), FakeRuntime)
        self.assertEqual(answer["index_update"], {"embedded": 1, "reused": 1})
        self.assertEqual(self.build()["embedded"], 0)

    def test_model_profile_and_vector_corruption_rejected(self):
        self.build()
        cache = worker.index_path(self.db_path)
        with closing(sqlite3.connect(cache)) as db, db:
            blob = np.full(768, np.nan, dtype="<f4").tobytes()
            db.execute("UPDATE embeddings SET vector=? WHERE node_id=1", (blob,))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            worker.search(self.request(), FakeRuntime)
        with closing(sqlite3.connect(cache)) as db, db:
            value = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
            value["profile"] = "different-model"
            db.execute("UPDATE metadata SET value=?", (json.dumps(value),))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            worker.search(self.request(), FakeRuntime)

    def test_paths_outside_checkout_rejected(self):
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("UPDATE nodes SET file_path='../outside.py' WHERE id=1")
        # An existing, explicit external path is needed to exercise containment itself.
        with tempfile.NamedTemporaryFile(dir=self.root.parent, suffix=".py", delete=False) as file:
            external = Path(file.name)
            file.write(b"def outside():\n    pass\n")
        try:
            with closing(sqlite3.connect(self.db_path)) as db, db:
                db.execute("UPDATE nodes SET file_path=? WHERE id=1", (str(external),))
            with self.assertRaisesRegex(ValueError, "leaves the repository"):
                worker.snapshot(self.db_path, "test")
        finally:
            external.unlink()

    def test_virtual_builtin_nodes_are_reported(self):
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("INSERT INTO nodes VALUES(3,'test','Method','builtins.get',"
                       "'<python-builtins>',1,2)")
        result = self.build()
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["excluded_virtual_nodes"], 1)

    def test_opened_handle_and_corpus_limits_are_verified(self):
        with patch.object(worker, "opened_path", return_value=self.root.parent / "outside.py"):
            with self.assertRaisesRegex(ValueError, "handle leaves"):
                worker.snapshot(self.db_path, "test")
        with patch.object(worker, "MAX_SOURCE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "source corpus"):
                self.build()
        with patch.object(worker, "MAX_DOCUMENT_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "document corpus"):
                self.build()
        self.assertEqual(FakeRuntime.calls, 0)

    def test_invalid_requests_and_empty_index(self):
        with self.assertRaisesRegex(ValueError, "missing/stale/incompatible"):
            worker.search(self.request(), FakeRuntime)
        for queries, limit in [([], 2), ([""], 2), (["alpha"], True),
                               (["alpha"], 100501), (["x" * 4097], 2)]:
            request = self.request()
            request.update(queries=queries, limit=limit)
            with self.assertRaisesRegex(ValueError, "contract"):
                worker.search(request, FakeRuntime)

    def test_refresh_rebinds_native_ids_and_removes_deleted_nodes(self):
        self.build()
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("UPDATE nodes SET id=id+10")
            db.execute("DELETE FROM nodes WHERE id=12")
            db.execute("UPDATE store_meta SET v='2' WHERE k='mutation_gen'")
        answer = worker.search(self.request(), FakeRuntime)
        self.assertEqual(answer["index_update"], {"embedded": 0, "reused": 1})
        self.assertEqual([r["node_id"] for r in answer["results"]], [11])

    def test_explicit_rebuild_repairs_wrong_cached_ids(self):
        self.build()
        with closing(sqlite3.connect(worker.index_path(self.db_path))) as db, db:
            db.execute("UPDATE embeddings SET node_id=99 WHERE node_id=1")
        with self.assertRaisesRegex(ValueError, "does not match"):
            worker.search(self.request(), FakeRuntime)
        result = self.build()
        self.assertEqual((result["embedded"], result["reused"]), (0, 2))
        self.assertEqual(worker.search(self.request(), FakeRuntime)["results"][0]["node_id"], 1)

    def test_update_lock_refuses_second_publisher_and_releases_after_close(self):
        self.build()
        before = worker.index_path(self.db_path).read_bytes()
        with closing(sqlite3.connect(str(worker.index_path(self.db_path)) + ".lock")) as lock:
            lock.execute("BEGIN IMMEDIATE")
            with self.assertRaisesRegex(ValueError, "busy"):
                self.build()
            self.assertEqual(worker.index_path(self.db_path).read_bytes(), before)
        self.assertEqual(self.build()["embedded"], 0)

    def test_duplicate_new_documents_are_encoded_once(self):
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("INSERT INTO nodes VALUES(3,'test','Function','test.alpha_variant',"
                       "'test.py',1,2)")
        result = self.build()
        self.assertEqual((result["embedded"], result["reused"]), (2, 1))
        self.assertEqual(FakeRuntime.calls, 2)

    def test_persistent_refresh_reuses_model_and_encodes_only_changed_function(self):
        self.build()
        loads = []
        def factory(allow_download):
            loads.append(allow_download)
            return FakeRuntime()
        with self.running_worker(factory) as directory:
            self.exchange(directory, self.request(), 1)
            self.source.write_text("def alpha():\n    return 8\n\ndef beta():\n    return 2\n")
            with closing(sqlite3.connect(self.db_path)) as db, db:
                db.execute("UPDATE file_hashes SET sha256=?",
                           (hashlib.sha256(self.source.read_bytes()).hexdigest(),))
                db.execute("UPDATE store_meta SET v='2' WHERE k='mutation_gen'")
            refreshed = self.exchange(directory, self.request(), 2)
            self.assertEqual(refreshed["index_update"], {"embedded": 1, "reused": 1})
            self.assertEqual(loads, [False])

    def test_corrupt_publisher_lock_is_not_reported_as_busy(self):
        lock = Path(str(worker.index_path(self.db_path)) + ".lock")
        lock.write_bytes(b"not a SQLite database" * 100)
        with self.assertRaises(sqlite3.DatabaseError) as caught:
            self.build()
        self.assertNotIn("busy", str(caught.exception))

    def test_wrong_manifest_generation_is_rebound_without_embedding(self):
        self.build()
        cache = worker.index_path(self.db_path)
        with closing(sqlite3.connect(cache)) as db, db:
            value = json.loads(db.execute("SELECT value FROM metadata").fetchone()[0])
            value["generation"] = "wrong"
            db.execute("UPDATE metadata SET value=?", (json.dumps(value),))
        result = worker.search(self.request(), FakeRuntime)
        self.assertEqual(result["index_update"], {"embedded": 0, "reused": 2})
        self.assertEqual(result["generation"], self.request()["generation"])

    def test_failed_rebuild_preserves_previous_database(self):
        self.build()
        before = worker.index_path(self.db_path).read_bytes()
        self.source.write_text("def alpha():\n    return 7\n\ndef beta():\n    return 2\n")
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute("UPDATE file_hashes SET sha256=?",
                       (hashlib.sha256(self.source.read_bytes()).hexdigest(),))
            db.execute("UPDATE store_meta SET v='2' WHERE k='mutation_gen'")

        class FailingRuntime(FakeRuntime):
            def encode(self, texts, query=False):
                raise RuntimeError("simulated inference failure")

        with self.assertRaisesRegex(RuntimeError, "simulated"):
            worker.build_index(self.db_path, "test", runtime_factory=FailingRuntime)
        self.assertEqual(worker.index_path(self.db_path).read_bytes(), before)
        self.assertFalse(list(self.root.glob("*.pending")))


if __name__ == "__main__":
    unittest.main()
