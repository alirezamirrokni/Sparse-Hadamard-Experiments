import contextlib
import hashlib
import importlib
import io
import os
import pickle
import subprocess
import sys
import sysconfig
import tarfile
import urllib.request
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

REPOSITORY = "andisheh94/Efficiently-Learning-Fourier-Sparse-Set-Functions"
REVISION = "4c3e161e13062a356afcba1f41d0493b0373336c"
CODE = Path(__file__).resolve().parents[1] / "data" / "originals" / "code"
MODULES = ["utils", "hashing", "WHT", "randomGraph", "dataGraph", "processGraphs", "decision_tree", "apgwrapper",
           "apgpy", "sparseWHT", "proximalMethod"]


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def prepare():
    if not CODE.exists():
        url = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{REVISION}"
        prefix = f"{REPOSITORY.split('/')[1]}-{REVISION}/"
        with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url, timeout=120).read())) as archive:
            members = [m for m in archive.getmembers() if m.name.startswith(prefix + "code/")]
            for member in members:
                member.name = member.name[len(prefix):]
            archive.extractall(CODE.parent, members=members, filter="data")
    native = CODE / "fastwht" / "python"
    if not (native / ("_hadamardKernel" + sysconfig.get_config_var("EXT_SUFFIX"))).exists():
        environment = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
        command = [sys.executable, "setup.py", "build_ext", "--inplace", "--swig-opts=-c++"]
        subprocess.run(command, cwd=native, env=environment, check=True, capture_output=True)


@lru_cache(maxsize=1)
def load():
    prepare()
    sys.path[:0] = [str(CODE), str(CODE / "apgpy-proximal")]
    modules = {name: importlib.import_module(name) for name in MODULES}
    return SimpleNamespace(
        swht=modules["sparseWHT"].SWHT,
        stobbe=modules["proximalMethod"].ProximalMethod,
        graph=modules["dataGraph"].DataGraph,
        get_graph=modules["processGraphs"].getGraph,
        wht=modules["WHT"].WHT,
    )


def read_source(name):
    return (CODE.parent / name).read_text()


@lru_cache(maxsize=128)
def graph_template(dimension, edges, snapshot):
    previous = Path.cwd()
    try:
        os.chdir(CODE)
        with contextlib.redirect_stdout(io.StringIO()):
            return load().get_graph(dimension, edges, snapshot)
    finally:
        os.chdir(previous)


def graph(spec):
    template = graph_template(spec["dimension"], spec["edges"], spec["snapshot"])
    return load().graph(spec["dimension"], dict(template.graph))


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Executable pickle content is prohibited: {module}.{name}")


def published_setting(spec, method):
    label = {"swht": "SWHT", "stobbe": "Stobbe"}[method]
    axis = "Edge" if spec["kind"] == "graph_edges" else "Vertex"
    filename = f"code/experimentResults/{label}{axis}FileSec{spec['snapshot']}.pkl"
    path = CODE.parent / filename
    records = DataOnlyUnpickler(io.BytesIO(path.read_bytes())).load()[spec["edges"] if axis == "Edge" else spec["dimension"]]
    eligible = [(index, row) for index, row in enumerate(records) if row[0] >= 0.9]
    if method == "swht":
        index, row = min(eligible, key=lambda item: item[1][2])
        settings = {"C": float(row[3]), "ratio": float(row[4]), "coordinate_bins": 3, "detection_iterations": 1}
        selection = "fastest reported mean at recovery >= 0.9"
    else:
        index, row = eligible[0]
        settings = {"C": [1.5, 2.0, 2.5, 3.0][index], "lambda": 1.0}
        selection = "first reported C at recovery >= 0.9"
    return {
        "settings": settings,
        "source": filename,
        "source_git_blob": git_blob(path),
        "published_recovery": float(row[0]),
        "published_mean_seconds": float(row[2]),
        "selection": selection,
    }
