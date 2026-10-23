import ast
from functools import lru_cache
import hashlib
import itertools
import json
import math
import pickle
import random
import statistics
import types
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .algebra import fwht
from .storage import atomic_text


MANIFEST = Path(__file__).with_name("upstream_manifest.json")


def fetch_sources(directory):
    manifest = json.loads(MANIFEST.read_text())
    directory = Path(directory)
    for item in manifest["files"]:
        destination = directory / item["path"]
        if destination.exists():
            data = destination.read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['revision']}/{urllib.parse.quote(item['path'])}"
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            destination.parent.mkdir(parents=True, exist_ok=True)
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != item["git_blob"]:
            raise ValueError(f"Upstream checksum mismatch: {item['path']}")
        if not destination.exists():
            atomic_text(destination, data.decode())
    atomic_text(directory / "manifest.json", json.dumps(manifest, indent=2))


def checked_source(directory, name):
    manifest = json.loads(MANIFEST.read_text())
    item = next(item for item in manifest["files"] if item["path"] == name)
    data = (Path(directory) / name).read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if blob != item["git_blob"]:
        raise ValueError(f"Upstream checksum mismatch: {name}")
    return data.decode()


@lru_cache(maxsize=512)
def graph_edges(directory, dimension, edge_count, snapshot):
    def read_edges(number):
        result = {}
        vertices = {}
        for line in checked_source(
            directory, f"code/graphData/{number}.txt"
        ).splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            a, b = map(int, line.split()[:2])
            vertices[a] = None
            vertices[b] = None
            result[tuple(sorted((a, b)))] = None
        return result, vertices

    first, first_vertices = read_edges(snapshot)
    second, second_vertices = read_edges(snapshot + 2)
    common = {
        v: i for i, v in enumerate(v for v in first_vertices if v in second_vertices)
    }
    changed = {}
    for source, other in [(first, second), (second, first)]:
        for a, b in source:
            if (a, b) not in other and a in common and b in common:
                changed[tuple(sorted((common[a], common[b])))] = 1
    if dimension > len(common):
        raise ValueError("Requested more vertices than the data contain")
    degrees = [0] * len(common)
    for a, b in changed:
        degrees[a] += 1
        degrees[b] += 1
    top = sorted(range(len(common)), key=lambda j: degrees[j])[-dimension:]
    mapping = {v: i for i, v in enumerate(top)}
    retained = {
        tuple(sorted((mapping[a], mapping[b]))): 1
        for a, b in changed
        if a in mapping and b in mapping
    }
    if edge_count >= 0:
        chosen = random.Random(1).sample(list(retained), edge_count)
        retained = {edge: 1 for edge in chosen}
    return tuple(sorted(retained))


def execute_definitions(source, namespace):
    tree = ast.parse(source)
    tree.body = [
        node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    exec(compile(tree, "<pinned-upstream>", "exec"), namespace)


def swht_recover(oracle, dimension, sparsity, settings, directory, seed):
    namespace = {
        "np": np,
        "ceil": math.ceil,
        "floor": math.floor,
        "log": math.log,
        "isclose": math.isclose,
        "product": itertools.product,
        "statistics": statistics,
        "pickle": pickle,
    }
    execute_definitions(checked_source(directory, "code/utils.py"), namespace)
    execute_definitions(checked_source(directory, "code/hashing.py"), namespace)

    class BatchedHashing(namespace["Hashing"]):
        def do_TimeHash(self, signal, shift):
            cache_key = tuple(np.asarray(shift).ravel().tolist())
            if cache_key in self.cache:
                return self.cache[cache_key]
            indices = np.arange(self.B, dtype=np.uint64)
            coordinates = (
                (
                    indices[:, None]
                    >> np.arange(self.b - 1, -1, -1, dtype=np.int64).astype(np.uint64)
                )
                & 1
            ).astype(np.uint8)
            points = ((coordinates @ self.P.T + np.asarray(shift).ravel()) % 2).astype(
                np.uint8
            )
            transformed = signal.evaluate(points).reshape(tuple([2] * self.b))
            self.cache[cache_key] = transformed
            return transformed

    namespace["hashing"] = types.SimpleNamespace(Hashing=BatchedHashing)
    namespace["utils"] = types.SimpleNamespace(inp=namespace["inp"])
    namespace["WHT"] = lambda array: fwht(array) / array.size
    execute_definitions(checked_source(directory, "code/sparseWHT.py"), namespace)

    class Signal:
        shape = tuple([2] * dimension)

        def __getitem__(self, point):
            return float(oracle(np.asarray(point, dtype=np.uint8)[None, :])[0])

        def evaluate(self, points):
            return oracle(points)

    state = np.random.get_state()
    np.random.seed(seed % (2**32))
    try:
        algorithm = namespace["SWHT"](
            dimension,
            sparsity,
            settings.get("C", 1.2),
            settings.get("ratio", 1.3),
            settings.get("coordinate_bins", 3),
            settings.get("detection_iterations", 1),
        )
        recovered = algorithm.run(Signal())
    finally:
        np.random.set_state(state)
    frequencies = [tuple(np.flatnonzero(f).tolist()) for f in recovered]
    return {
        "frequencies": frequencies,
        "values": [float(v) for v in recovered.values()],
        "status": "ok",
        "rounds": 1,
        "implementation": "pinned upstream SWHT with portable normalized WHT",
    }
