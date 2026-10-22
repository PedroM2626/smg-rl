"""Audit installed packages, normalizing CUDA's local version suffix for PyPI."""

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path


def audit(auditor, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    requirements = output.with_suffix(".requirements.txt")
    versions = {distribution.metadata["Name"]: distribution.version.split("+", 1)[0]
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"].lower().replace("_", "-") != "smg-rl"}
    requirements.write_text("\n".join(f"{name}=={version}" for name, version in sorted(versions.items()))
                            + "\n", encoding="utf-8")
    return subprocess.run([str(auditor), "-m", "pip_audit", "--no-deps", "--disable-pip",
                           "-r", str(requirements), "--format", "json", "--output", str(output)],
                          check=False).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditor", default=sys.executable,
                        help="Python interpreter with pip-audit installed; can be an isolated venv.")
    parser.add_argument("--output", default=".local/dependency-audit.json")
    args = parser.parse_args()
    raise SystemExit(audit(args.auditor, args.output))
