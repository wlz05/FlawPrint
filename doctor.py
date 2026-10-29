# -*- coding: utf-8 -*-
# FlawPrint: Cross-Domain Multimodal Fake News Detection via Copula Residuals of Physical Imperfection Traces

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
from os.path import abspath
from pathlib import Path, PureWindowsPath
import platform
import re
import sys

CORE = {"numpy": "numpy", "scipy": "scipy", "Pillow": "PIL", "scikit-learn": "sklearn", "requests": "requests", "pypdf": "pypdf"}
TABLES = {"pandas": "pandas", "openpyxl": "openpyxl", "pyarrow": "pyarrow", "xlrd": "xlrd"}
SEMANTIC = {"torch": "torch", "transformers": "transformers"}
LIMITS = {"numpy": ((1, 26), (3, 0)), "scipy": ((1, 11), (2, 0)), "Pillow": ((10, 0), (13, 0)), "scikit-learn": ((1, 4), (2, 0)), "requests": ((2, 31), (3, 0)), "pypdf": ((4, 0), (7, 0)), "pandas": ((2, 2), (4, 0)), "openpyxl": ((3, 1), (4, 0)), "pyarrow": ((15, 0), (30, 0)), "xlrd": ((2, 0), (3, 0)), "torch": ((2, 2), None), "transformers": ((4, 40), (6, 0))}

def compatible(name: str, version: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        return False
    number = tuple(map(int, match.groups()))
    lower, upper = LIMITS[name]
    return number >= lower and (upper is None or number < upper)

def environment(tables: bool = False, semantic: bool = False) -> dict:
    selected = dict(CORE)
    if tables:
        selected.update(TABLES)
    if semantic:
        selected.update(SEMANTIC)
    versions, unavailable, incompatible = {}, [], []
    for distribution, module in selected.items():
        try:
            present = importlib.util.find_spec(module) is not None
            versions[distribution] = importlib.metadata.version(distribution) if present else None
        except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
            versions[distribution] = None
        if versions[distribution] is None:
            unavailable.append(distribution)
        elif not compatible(distribution, versions[distribution]):
            incompatible.append(distribution)
    root_ok = Path("./main.py").is_file() and Path("./engine/package.py").is_file()
    supported = sys.version_info >= (3, 10)
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "implementation": platform.python_implementation(), "system": platform.system(), "machine": platform.machine(), "packages": versions, "missing": unavailable, "incompatible": incompatible, "python_supported": supported, "workspace_ready": root_ok, "passed": supported and root_ok and not unavailable and not incompatible, "path_base": "process working directory", "installation": 'python -m pip install "numpy>=1.26,<3" "scipy>=1.11,<2" "Pillow>=10,<13" "scikit-learn>=1.4,<2" "requests>=2.31,<3" "pypdf>=4,<7"', "optional_tables": 'python -m pip install "pandas>=2.2,<4" "openpyxl>=3.1,<4" "pyarrow>=15,<30" "xlrd>=2,<3"', "optional_semantic": 'python -m pip install "torch>=2.2" "transformers>=4.40,<6"'}

def relative(value: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("Use a nonempty relative output path")
    text = value.replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or PureWindowsPath(text).drive or text.startswith("/"):
        raise ValueError("Use a relative output path")
    if text == "~" or text.startswith("~/") or text.casefold().startswith(("http://", "https://", "file://")):
        raise ValueError("Use a relative output path")
    return path

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the reproduction environment using only the Python standard library.")
    parser.add_argument("--tables", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--output", default="./outputs/environment.json")
    parser.add_argument("--freeze", help="Write installed dependency versions to a relative requirements file")
    args = parser.parse_args()
    try:
        output = relative(args.output)
        freeze = relative(args.freeze) if args.freeze else None
        if freeze and Path(abspath(freeze)).resolve() == Path(abspath(output)).resolve():
            raise ValueError("Environment JSON and version list must be distinct")
        result = environment(args.tables, args.semantic)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        if freeze:
            freeze.parent.mkdir(parents=True, exist_ok=True)
            freeze.write_text("\n".join(f"{name}=={version}" for name, version in result["packages"].items() if version is not None) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=True))
        raise SystemExit(0 if result["passed"] else 1)
    except (ValueError, OSError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    main()

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
