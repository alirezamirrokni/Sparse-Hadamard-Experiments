import itertools
import time
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm

from .algebra import bits, degree_dictionary, top_indices, tuple_characters
from .recovery import prefix_recover
from .operators import DegreeOperator
from .signals import random_signal
from .solvers import solve
from .storage import BudgetExceeded, Oracle, digest


class DigitsTask:
    active_dimension = 12
    key_dimension = 12
    batch_size = 1

    def __init__(self, seed):
        data = load_digits()
        train_x, test_x, train_y, test_y = train_test_split(
            data.data,
            data.target,
            test_size=0.2,
            stratify=data.target,
            random_state=seed,
        )
        train_x, valid_x, train_y, valid_y = train_test_split(
            train_x, train_y, test_size=0.25, stratify=train_y, random_state=seed + 1
        )
        self.train_x, self.valid_x, self.test_x = train_x, valid_x, test_x
        self.train_y, self.valid_y, self.test_y = train_y, valid_y, test_y
        self.seed = seed

    def model(self, point):
        b = [int(v) for v in point[:12]]
        alpha = [1e-6, 1e-5, 1e-4, 1e-3][b[1] + 2 * b[2]]
        rate = [0.001, 0.01, 0.05, 0.2][b[4] + 2 * b[5]]
        estimator = SGDClassifier(
            loss=["hinge", "log_loss"][b[0]],
            alpha=alpha,
            penalty=["l2", "elasticnet"][b[3]],
            learning_rate=["constant", "adaptive"][b[6]],
            eta0=rate,
            average=bool(b[7]),
            fit_intercept=bool(b[8]),
            class_weight=[None, "balanced"][b[9]],
            max_iter=[25, 75][b[10]],
            tol=None,
            random_state=self.seed,
            n_jobs=1,
        )
        scaler = [StandardScaler(), MinMaxScaler()][b[11]]
        return make_pipeline(scaler, estimator)

    def __call__(self, points):
        output = []
        for point in points:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = self.model(point)
                model.fit(self.train_x, self.train_y)
            output.append(1 - accuracy_score(self.valid_y, model.predict(self.valid_x)))
        return np.asarray(output)

    def test(self, point):
        model = self.model(point)
        model.fit(
            np.concatenate([self.train_x, self.valid_x]),
            np.concatenate([self.train_y, self.valid_y]),
        )
        return float(1 - accuracy_score(self.test_y, model.predict(self.test_x)))


class HierarchicalTask:
    def __init__(self, dimension, seed, noise=0.0):
        self.dimension = dimension
        self.seed = seed
        self.noise = noise
        rng = np.random.default_rng(seed)
        self.functions = []
        for level in range(3):
            functions = []
            for node in range(32**level):
                frequencies = []
                for _ in range(5):
                    degree = int(rng.integers(1, min(3, dimension) + 1))
                    frequencies.append(
                        tuple(
                            sorted(
                                rng.choice(dimension, degree, replace=False).tolist()
                            )
                        )
                    )
                weights = rng.uniform(10 + 10 ** (-level), 10 + 10 ** (2 - level), 5)
                functions.append((frequencies, weights))
            self.functions.append(functions)

    def __call__(self, points):
        result = np.zeros(len(points))
        nodes = np.zeros(len(points), dtype=int)
        for level in range(3):
            children = np.zeros(len(points), dtype=int)
            for node in np.unique(nodes):
                positions = np.flatnonzero(nodes == node)
                frequencies, weights = self.functions[level][int(node)]
                parities = tuple_characters(points[positions], frequencies)
                result[positions] += parities @ weights
                children[positions] = (parities < 0).astype(int) @ (1 << np.arange(5))
            nodes = nodes * 32 + children
        if self.noise:
            for i, point in enumerate(points):
                seed = int(digest([self.seed, point.tolist()])[:16], 16)
                result[i] += np.random.default_rng(seed).uniform(
                    -self.noise, self.noise
                )
        return result


def make_task(spec, seed):
    kind = spec["task"]
    if kind == "digits":
        return DigitsTask(seed)
    if kind == "hierarchical":
        return HierarchicalTask(spec["dimension"], seed, spec.get("noise", 0.0))
    signal = random_signal(
        spec["dimension"],
        spec.get("sparsity", 8),
        seed,
        "gaussian",
        degree=spec.get("degree", 3),
    )
    if not spec.get("noise", 0.0):
        return signal

    class NoisySparse:
        def __call__(self, points):
            values = signal(points)
            for i, point in enumerate(points):
                noise_seed = int(digest([seed, point.tolist()])[:16], 16)
                values[i] += np.random.default_rng(noise_seed).uniform(
                    -spec["noise"], spec["noise"]
                )
            return values

    return NoisySparse()


def candidate_minimizers(frequencies, values, dimension, rng, count=1):
    active = sorted(set(itertools.chain.from_iterable(frequencies)))
    if not active:
        return np.zeros((count, dimension), dtype=np.uint8), []
    positions = {coordinate: j for j, coordinate in enumerate(active)}
    reduced = [tuple(positions[j] for j in f) for f in frequencies]
    if len(active) <= 18:
        pool = bits(np.arange(1 << len(active), dtype=np.uint64), len(active))
    else:
        pool = rng.integers(0, 2, (8192, len(active)), dtype=np.uint8)
        for _ in range(4):
            for coordinate in rng.permutation(len(active)):
                before = tuple_characters(pool, reduced) @ values
                pool[:, coordinate] ^= 1
                after = tuple_characters(pool, reduced) @ values
                pool[after > before, coordinate] ^= 1
    predicted = tuple_characters(pool, reduced) @ values
    selected = np.argsort(predicted, kind="stable")[:count]
    result = np.zeros((len(selected), dimension), dtype=np.uint8)
    result[:, active] = pool[selected]
    return result, active


def run_hpo(spec, method, settings, store, key, seed, implementation=None):
    dimension = spec["dimension"]
    objective = make_task(spec, seed)
    effective_dimension = getattr(objective, "key_dimension", dimension)
    if spec["budget"] > 1 << effective_dimension:
        raise ValueError("HPO budget exceeds the number of effective configurations")
    identity = {"task": spec["task"]}
    if spec["task"] != "digits":
        identity.update(
            {
                name: spec.get(name)
                for name in ["dimension", "sparsity", "degree", "noise"]
            }
        )
    namespace = digest(
        {"task": identity, "seed": seed, "implementation": implementation}
    )
    oracle = Oracle(objective, dimension, store, namespace, budget=spec["budget"])
    rng = np.random.default_rng(seed + 937)
    fixed = {}
    fit_seconds = 0.0
    optimization_seconds = 0.0
    stage_records = []
    trace = []
    best_value = float("inf")
    best_point = None
    started = time.perf_counter()
    original_call = oracle.__call__

    def observe(points):
        nonlocal best_value, best_point
        points = np.atleast_2d(points)
        values = original_call(points)
        for point, value in zip(points, values):
            if value < best_value:
                best_value = float(value)
                best_point = point.copy()
        return values

    class Observed:
        def __call__(self, points):
            return observe(points)

        def __getattr__(self, name):
            return getattr(oracle, name)

    observed = Observed()
    stage_count = spec.get("stages", 3)
    if method in {"random", "gp_ei"}:
        stage_count = 0
    if method == "gp_ei":
        observations_x = []
        observations_y = []
        saved_gp = store.get(f"{key}/gp-state")
        if saved_gp is not None:
            oracle.restore(saved_gp["oracle_state"])
            observations_x = saved_gp["observations_x"]
            observations_y = saved_gp["observations_y"]
            best_value = saved_gp["best_value"]
            best_point = np.asarray(saved_gp["best_point"], dtype=np.uint8)
            optimization_seconds = saved_gp["optimization_seconds"]
            stage_records = saved_gp["stages"]
            rng.bit_generator.state = saved_gp["rng"]
        while oracle.count < spec["budget"]:
            optimize_started = time.perf_counter()
            pool = rng.integers(0, 2, (1024, dimension), dtype=np.uint8)
            if len(observations_y) >= 8:
                model = GaussianProcessRegressor(
                    kernel=Matern(length_scale=np.sqrt(dimension), nu=2.5),
                    alpha=1e-5,
                    optimizer=None,
                    normalize_y=True,
                    random_state=seed,
                )
                model.fit(np.asarray(observations_x), np.asarray(observations_y))
                mean, deviation = model.predict(pool, return_std=True)
                improvement = min(observations_y) - mean - 0.001
                z = improvement / np.maximum(deviation, 1e-12)
                acquisition = improvement * norm.cdf(z) + deviation * norm.pdf(z)
                key_dimension = getattr(objective, "key_dimension", dimension)
                keys = np.packbits(pool[:, :key_dimension], axis=1, bitorder="little")
                used = np.asarray([row.tobytes() in oracle.seen for row in keys])
                acquisition[used] = -np.inf
                point = pool[[int(np.argmax(acquisition))]]
            else:
                point = pool[:1]
            optimization_seconds += time.perf_counter() - optimize_started
            value = float(observe(point)[0])
            observations_x.append(point[0])
            observations_y.append(value)
            stage_records.append(
                {
                    "stage": len(stage_records) + 1,
                    "queries": oracle.count,
                    "fit_seconds": 0.0,
                    "optimization_seconds": optimization_seconds
                    - sum(r["optimization_seconds"] for r in stage_records),
                }
            )
            store.put(
                f"{key}/gp-state",
                {
                    "oracle_state": oracle.snapshot(),
                    "observations_x": observations_x,
                    "observations_y": observations_y,
                    "best_value": best_value,
                    "best_point": best_point,
                    "optimization_seconds": optimization_seconds,
                    "stages": stage_records,
                    "rng": rng.bit_generator.state,
                },
            )
    for stage in range(stage_count):
        rng = np.random.default_rng(seed + 937 + stage * 100003)
        state_key = f"{key}/hpo-stage/{stage}"
        saved = store.get(state_key)
        if saved is not None:
            oracle.restore(saved["oracle_state"])
            fixed = {int(k): int(v) for k, v in saved["fixed"].items()}
            best_value = saved["best_value"]
            best_point = np.asarray(saved["best_point"], dtype=np.uint8)
            fit_seconds += saved["fit_seconds"]
            optimization_seconds += saved["optimization_seconds"]
            stage_records.append(saved["report"])
            continue
        free = [j for j in range(dimension) if j not in fixed]
        if not free:
            break

        def restriction(points):
            full = np.zeros((len(points), dimension), dtype=np.uint8)
            full[:, free] = points
            for j, value in fixed.items():
                full[:, j] = value
            return observed(full)

        remaining = spec["budget"] - oracle.count
        training_budget = min(
            spec.get("samples_per_stage", 100),
            max(
                1,
                (remaining - spec.get("final_samples", 32))
                // max(1, stage_count - stage),
            ),
        )
        stage_started = time.perf_counter()
        evaluation_before = oracle.actual_evaluation_seconds
        if method.startswith("prefix"):
            local = dict(settings)
            local.setdefault("solver", "fista")
            local["validated"] = False
            local["retention"] = 1

            class Restricted:
                budget = training_budget
                actual_evaluation_seconds = 0.0

                def __init__(self):
                    self.seen = set()
                    self.requests = 0

                @property
                def count(self):
                    return len(self.seen)

                def __call__(self, points):
                    keys = {
                        row.tobytes()
                        for row in np.packbits(points, axis=1, bitorder="little")
                    }
                    if len(self.seen | keys) > self.budget:
                        raise BudgetExceeded("HPO stage query budget exceeded")
                    self.seen.update(keys)
                    self.requests += len(points)
                    before = oracle.actual_evaluation_seconds
                    result = restriction(points)
                    self.actual_evaluation_seconds += (
                        oracle.actual_evaluation_seconds - before
                    )
                    return result

                def snapshot(self):
                    return {
                        "seen": sorted(v.hex() for v in self.seen),
                        "requests": self.requests,
                        "parent": oracle.snapshot(),
                        "best_value": best_value,
                        "best_point": None
                        if best_point is None
                        else best_point.tolist(),
                    }

                def restore(self, state):
                    nonlocal best_value, best_point
                    self.seen = {bytes.fromhex(v) for v in state["seen"]}
                    self.requests = state["requests"]
                    oracle.restore(state["parent"])
                    best_value = state["best_value"]
                    best_point = np.asarray(state["best_point"], dtype=np.uint8)

            restricted = Restricted()
            recovered = prefix_recover(
                restricted,
                len(free),
                spec.get("sparsity", 5),
                local,
                store,
                f"{key}/recovery/{stage}",
                seed + 991 * stage,
                degree=min(spec.get("degree", 3), len(free)),
            )
            frequencies = [tuple(f) for f in recovered["frequencies"]]
            values = np.asarray(recovered["values"])
        else:
            points = rng.integers(0, 2, (training_budget, len(free)), dtype=np.uint8)
            response = restriction(points) / np.sqrt(training_budget)
            dictionary = degree_dictionary(
                len(free),
                min(spec.get("degree", 3), len(free)),
                settings.get("max_global_columns", 2_000_000),
            )
            operator = DegreeOperator(
                points,
                dictionary,
                max_cache_bytes=settings.get("max_dense_bytes", 268_435_456) // 2,
            )
            fitting = dict(settings)
            fitting["solver"] = "fista"
            fitted, report = solve(
                operator, response, fitting, store, f"{key}/lasso/{stage}"
            )
            selected = top_indices(fitted, spec.get("sparsity", 5))
            frequencies = [dictionary[j] for j in selected]
            values = fitted[selected]
        stage_fit = time.perf_counter() - stage_started
        if method.startswith("prefix"):
            stage_fit = (
                sum(
                    max(0.0, item["seconds"] - item.get("evaluation_wall_seconds", 0.0))
                    for item in recovered["stages"]
                )
                + recovered.get("preprocessing_seconds", 0.0)
                + recovered.get("decoding_seconds", 0.0)
            )
        else:
            stage_fit = max(
                0.0, stage_fit - (oracle.actual_evaluation_seconds - evaluation_before)
            ) + report.get("resumed_seconds", 0.0)
        fit_seconds += stage_fit
        optimize_started = time.perf_counter()
        optimization_evaluation_before = oracle.actual_evaluation_seconds
        proposals, active = candidate_minimizers(frequencies, values, len(free), rng)
        if len(proposals):
            restriction(proposals[:1])
            for coordinate in active:
                fixed[free[coordinate]] = int(proposals[0, coordinate])
        stage_optimize = max(
            0.0,
            time.perf_counter()
            - optimize_started
            - (oracle.actual_evaluation_seconds - optimization_evaluation_before),
        )
        optimization_seconds += stage_optimize
        report = {
            "stage": stage + 1,
            "fixed_variables": len(fixed),
            "queries": oracle.count,
            "best_validation": best_value,
            "fit_seconds": stage_fit,
            "optimization_seconds": stage_optimize,
        }
        stage_records.append(report)
        trace.append(
            [
                oracle.count,
                best_value,
                oracle.evaluation_seconds + fit_seconds + optimization_seconds,
            ]
        )
        store.put(
            state_key,
            {
                "oracle_state": oracle.snapshot(),
                "fixed": fixed,
                "best_value": best_value,
                "best_point": best_point.tolist(),
                "fit_seconds": stage_fit,
                "optimization_seconds": stage_optimize,
                "report": report,
            },
        )
    rng = np.random.default_rng(seed + 7777)
    stagnant_queries = 0
    restriction_resets = 0
    while oracle.count < spec["budget"]:
        point = rng.integers(0, 2, (1, dimension), dtype=np.uint8)
        for coordinate, value in fixed.items():
            point[:, coordinate] = value
        before = oracle.count
        observe(point)
        trace.append(
            [
                oracle.count,
                best_value,
                oracle.evaluation_seconds + fit_seconds + optimization_seconds,
            ]
        )
        if oracle.count == before:
            stagnant_queries += 1
            if (
                stagnant_queries >= 32
                or len(fixed) == dimension
                or len(oracle.seen) >= 1 << min(20, dimension - len(fixed))
            ):
                fixed = {}
                stagnant_queries = 0
                restriction_resets += 1
        else:
            stagnant_queries = 0
    incumbent = float("inf")
    trace = []
    for count, value, evaluation_seconds in oracle.history:
        incumbent = min(incumbent, value)
        overhead = sum(
            record["fit_seconds"] + record["optimization_seconds"]
            for record in stage_records
            if record["queries"] <= count
        )
        trace.append([count, incumbent, evaluation_seconds + overhead])
    output = {
        "status": "ok",
        "best_validation": best_value,
        "queries": oracle.count,
        "sampled_rows": oracle.requests,
        "oracle_seconds": oracle.evaluation_seconds,
        "fit_seconds": fit_seconds,
        "optimization_seconds": optimization_seconds,
        "wall_seconds": time.perf_counter() - started,
        "trace": trace,
        "query_history": oracle.history,
        "stages": stage_records,
        "best_point": best_point.tolist(),
        "restriction_resets": restriction_resets,
    }
    if spec["task"] == "digits":
        output["test_error"] = objective.test(best_point)
    return output
