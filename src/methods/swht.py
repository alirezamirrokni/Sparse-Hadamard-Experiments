import ast
import itertools
import math
import pickle
import statistics
import types

import numpy as np

from .. import originals
from ..algebra import fwht
from ..originals import read_source as checked_source


def recover_graph(oracle, dimension, edges, settings):
    algorithm = originals.load().swht(
        dimension,
        edges,
        settings["C"],
        settings["ratio"],
        settings["coordinate_bins"],
        settings["detection_iterations"],
    )
    return algorithm.run(oracle)


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
