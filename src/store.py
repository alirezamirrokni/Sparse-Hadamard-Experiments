import hashlib
import io
import json
import sqlite3
import time
import zlib
from pathlib import Path

import numpy as np


def canonical(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=convert, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            "CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, kind TEXT, value BLOB);"
            "CREATE TABLE IF NOT EXISTS queries (namespace TEXT, point BLOB, value REAL, seconds REAL,"
            " PRIMARY KEY(namespace, point));"
        )

    def get(self, key):
        row = self.db.execute("SELECT value FROM records WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(zlib.decompress(row[0]))

    def put(self, key, value, kind="checkpoint"):
        data = zlib.compress(canonical(value).encode())
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (key, kind, data))

    def finish(self, key, value):
        data = zlib.compress(canonical(value).encode())
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (key, "result", data))
            self.db.execute("DELETE FROM records WHERE key LIKE ?", (key + "/%",))

    def keys(self, kind="result"):
        return {key for (key,) in self.db.execute("SELECT key FROM records WHERE kind=?", (kind,))}

    def iter_records(self, kind):
        for key, value in self.db.execute("SELECT key,value FROM records WHERE kind=? ORDER BY key", (kind,)):
            yield key, json.loads(zlib.decompress(value))

    def memo(self, key, function):
        value = self.get(key)
        if value is not None:
            return value
        started = time.perf_counter()
        value = function()
        value["seconds"] = time.perf_counter() - started + value.get("resumed_seconds", 0.0)
        self.put(key, value)
        return value

    def save_arrays(self, key, **arrays):
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (key, "arrays", stream.getvalue()))

    def load_arrays(self, key):
        row = self.db.execute("SELECT value FROM records WHERE key=? AND kind='arrays'", (key,)).fetchone()
        if row is None:
            return None
        with np.load(io.BytesIO(row[0]), allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    def close(self):
        self.db.close()


class BudgetExceeded(RuntimeError):
    pass


class Oracle:
    def __init__(self, function, dimension, store, namespace, budget=None):
        self.function = function
        self.dimension = dimension
        self.store = store
        self.namespace = namespace
        self.budget = budget
        self.seen = set()
        self.requests = 0
        self.evaluation_seconds = 0.0
        self.actual_evaluation_seconds = 0.0
        self.history = []

    def __call__(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=np.uint8))
        if points.shape[1] != self.dimension:
            raise ValueError("Oracle query has wrong dimension")
        key_dimension = getattr(self.function, "key_dimension", self.dimension)
        packed = np.packbits(points[:, :key_dimension], axis=1, bitorder="little")
        keys = [row.tobytes() for row in packed]
        self.requests += len(points)
        if self.budget is not None and len(self.seen) + len(set(keys) - self.seen) > self.budget:
            raise BudgetExceeded(f"Unique query budget {self.budget} exceeded")
        values, missing = {}, {}
        for i, key in enumerate(keys):
            if key in values or key in missing:
                continue
            row = self.store.db.execute(
                "SELECT value,seconds FROM queries WHERE namespace=? AND point=?", (self.namespace, key)
            ).fetchone()
            if row is None:
                missing[key] = i
            else:
                values[key] = row
        entries = list(missing.items())
        batch_size = getattr(self.function, "batch_size", 128)
        for start in range(0, len(entries), batch_size):
            batch = entries[start : start + batch_size]
            started = time.perf_counter()
            evaluated = np.asarray(self.function(points[[position for _, position in batch]]), dtype=float)
            duration = (time.perf_counter() - started) / len(batch)
            self.actual_evaluation_seconds += duration * len(batch)
            if not np.all(np.isfinite(evaluated)):
                raise ValueError("Nonfinite oracle response")
            with self.store.db:
                self.store.db.executemany(
                    "INSERT OR IGNORE INTO queries VALUES (?,?,?,?)",
                    [(self.namespace, key, float(value), duration) for (key, _), value in zip(batch, evaluated)],
                )
            values.update({key: (float(value), duration) for (key, _), value in zip(batch, evaluated)})
        for key in keys:
            if key not in self.seen:
                self.seen.add(key)
                self.evaluation_seconds += values[key][1]
                self.history.append([len(self.seen), values[key][0], self.evaluation_seconds])
        return np.asarray([values[key][0] for key in keys])

    @property
    def count(self):
        return len(self.seen)

    def snapshot(self):
        return {
            "seen": sorted(key.hex() for key in self.seen),
            "requests": self.requests,
            "evaluation_seconds": self.evaluation_seconds,
            "history": self.history,
        }

    def restore(self, state):
        self.seen = {bytes.fromhex(key) for key in state["seen"]}
        self.requests = state["requests"]
        self.evaluation_seconds = state["evaluation_seconds"]
        self.history = state["history"]
