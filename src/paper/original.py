import contextlib
import hashlib
import importlib
import io
import json
import os
import pickle
import subprocess
import sys
import sysconfig
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("original_manifest.json")
DIRECTORY = ROOT / "data" / "originals"


def verify(path, expected):
    data = path.read_bytes()
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Original source checksum mismatch: {path}")
    return data


def prepare():
    manifest = json.loads(MANIFEST.read_text())

    def fetch(item):
        path = DIRECTORY / item["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            existing = ROOT / "data" / "upstream" / item["path"]
            if existing.exists():
                data = verify(existing, item["git_blob"])
            else:
                url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['revision']}/{item['path']}"
                data = urllib.request.urlopen(url, timeout=120).read()
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(data)
            verify(temporary, item["git_blob"])
            temporary.replace(path)
        verify(path, item["git_blob"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(fetch, manifest["files"]))
    native = DIRECTORY / "code" / "fastwht" / "python"
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    extension = native / ("_hadamardKernel" + suffix)
    if not extension.exists():
        include = Path(sysconfig.get_paths()["include"])
        extra = []
        if not (include / "Python.h").exists():
            headers = DIRECTORY / "build-headers"
            extracted = headers / "unpacked"
            relative = (
                Path("usr/include")
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
            )
            if not (extracted / relative / "Python.h").exists():
                headers.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "apt-get",
                        "download",
                        f"libpython{sys.version_info.major}.{sys.version_info.minor}-dev",
                    ],
                    cwd=headers,
                    check=True,
                )
                for package in headers.glob("*.deb"):
                    subprocess.run(
                        ["dpkg-deb", "-x", str(package), str(extracted)], check=True
                    )
            extra = [
                "--include-dirs="
                + os.pathsep.join(
                    [str(extracted / relative), str(extracted / "usr/include")]
                )
            ]
        environment = os.environ.copy()
        environment["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
        )
        with (native / "build.log").open("w") as stream:
            subprocess.run(
                [
                    sys.executable,
                    "setup.py",
                    "build_ext",
                    "--force",
                    "--inplace",
                    "--swig-opts=-c++",
                    *extra,
                ],
                cwd=native,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    return manifest


@lru_cache(maxsize=1)
def load():
    prepare()
    code = DIRECTORY / "code"
    sys.path[:0] = [str(code), str(code / "apgpy-proximal")]
    names = [
        "utils",
        "hashing",
        "WHT",
        "randomGraph",
        "dataGraph",
        "processGraphs",
        "decision_tree",
        "apgwrapper",
        "apgpy",
        "sparseWHT",
        "proximalMethod",
    ]
    modules = {}
    for name in names:
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(code):
            raise RuntimeError(
                f"Unexpected module shadows original implementation: {name}"
            )
        modules[name] = module
    return SimpleNamespace(
        swht=modules["sparseWHT"].SWHT,
        stobbe=modules["proximalMethod"].ProximalMethod,
        graph=modules["dataGraph"].DataGraph,
        get_graph=modules["processGraphs"].getGraph,
        wht=modules["WHT"].WHT,
    )


@lru_cache(maxsize=128)
def graph_template(dimension, edges, snapshot):
    original = load()
    previous = Path.cwd()
    try:
        os.chdir(DIRECTORY / "code")
        with contextlib.redirect_stdout(io.StringIO()):
            return original.get_graph(dimension, edges, snapshot)
    finally:
        os.chdir(previous)


def graph(spec):
    template = graph_template(spec["dimension"], spec["edges"], spec["snapshot"])
    return load().graph(spec["dimension"], dict(template.graph))


class DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"Executable pickle content is prohibited: {module}.{name}"
        )


def published_setting(spec, method):
    if method not in {"swht", "stobbe"}:
        raise ValueError(method)
    label = "SWHT" if method == "swht" else "Stobbe"
    axis = "Edge" if spec["kind"] == "graph_edges" else "Vertex"
    key = spec["edges"] if axis == "Edge" else spec["dimension"]
    filename = f"code/experimentResults/{label}{axis}FileSec{spec['snapshot']}.pkl"
    manifest = json.loads(MANIFEST.read_text())
    entry = next(item for item in manifest["files"] if item["path"] == filename)
    payload = verify(DIRECTORY / filename, entry["git_blob"])
    table = DataOnlyUnpickler(io.BytesIO(payload)).load()
    records = table[key]
    eligible = [(index, row) for index, row in enumerate(records) if row[0] >= 0.9]
    if not eligible:
        raise RuntimeError(
            f"No published setting reaches 90% recovery: {filename}, {key}"
        )
    if method == "swht":
        index, row = min(eligible, key=lambda item: item[1][2])
        settings = {
            "C": float(row[3]),
            "ratio": float(row[4]),
            "coordinate_bins": 3,
            "detection_iterations": 1,
        }
    else:
        index, row = eligible[0]
        settings = {"C": [1.5, 2.0, 2.5, 3.0][index], "lambda": 1.0}
    return {
        "settings": settings,
        "source": filename,
        "source_git_blob": entry["git_blob"],
        "published_recovery": float(row[0]),
        "published_mean_seconds": float(row[2]),
        "selection": "fastest reported mean at recovery >= 0.9"
        if method == "swht"
        else "first reported C at recovery >= 0.9",
    }
