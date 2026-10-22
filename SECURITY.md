# Security policy

Report suspected vulnerabilities privately to the repository owner through
GitHub's private vulnerability reporting, when available. Avoid attaching ROMs,
savestates, credentials or private checkpoints to public issues.

## Trust boundaries

This is a local research application. It launches the configured Dolphin binary,
reads that process's mapped game RAM, and trains models from local assets. It is
not a sandbox for untrusted executables, Python packages, model archives or
configuration files. Run it under your normal user account without elevation.

SB3 model metadata can contain cloudpickled Python objects. **Load only trusted
local checkpoints.** SB3 2.9 loads tensor payloads with `weights_only=True`, and
the project does the same for curiosity state. This does not make arbitrary
SB3 ZIP metadata safe. Pair SHA-256 hashes detect corruption/mismatched files;
they do not authenticate an author. Never treat a supplied hash as permission
to load an unknown model.

Each worker uses a random local shared-memory mapping and private Windows events,
with sequence, version, PID, pause and frame-completion checks. Local same-user
processes are within the trust boundary; the bridge is not an isolation boundary
against a compromised account. Game memory is opened with read-only permissions.

The DSU server binds only to `127.0.0.1`, validates packet magic/version/length/CRC,
bounds registered peers, and handles disconnected clients. CRC is an error check,
not authentication. The native build still uses DSU device discovery; removing
the server without replacing discovery would disable its controls. Dolphin's own
network functionality is upstream functionality and is not firewalled by this
project. Analytics are disabled in owned worker profiles.

## Recovery and dependency maintenance

New checkpoint pairs are staged with an incomplete-publication marker, then
published with a checksum manifest. Recovery rejects interrupted or damaged
pairs. Existing local legacy pairs remain readable after ZIP integrity checks.
Measured worker receives and shutdowns are bounded, and the supervisor stops
after eight consecutive attempts with no new checkpoint instead of retrying a
permanent failure forever. Diagnostics identify that failed state.

The October 22, 2026 audit upgraded pip to 26.2.2, Torch to 2.14.1+cu126, and
SB3/sb3-contrib to 2.9.0. The tested CUDA setup is pinned in
`requirements-gpu.lock.txt`. Current project metadata requires those runtime
security updates. See [the revision evidence](docs/optimization-v4.md).

To repeat an installed-environment advisory audit without adding audit tooling
to the training environment:

```powershell
.\.venv\Scripts\python.exe -m venv .local/audit-env
.\.local\audit-env\Scripts\python.exe -m pip install --upgrade pip pip-audit
.\.venv\Scripts\python.exe scripts/audit_dependencies.py --auditor .local/audit-env/Scripts/python.exe --output .local/dependency-audit.json
```

The script excludes the local project from PyPI lookup and removes the CUDA
local-version suffix when querying advisories. Otherwise a tool can silently
skip `torch==...+cu126`. It audits exact installed versions, not the emulator
binary or the project's own code. Advisory data and an audit's coverage change
over time. A clean result is not a guarantee that all vulnerabilities are absent.
