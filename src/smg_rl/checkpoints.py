"""Publish policy/curiosity pairs transactionally and verify their integrity.

Checksums detect corruption and mismatched files, not malicious authors. SB3
archives contain serialized Python objects: only load locally trusted models.
"""

import hashlib
import json
import uuid
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_pair(model, intrinsic, base):
    # Supervisors only verify archives/hashes; avoid loading the ML runtime there.
    import torch

    base = Path(base)
    policy = base.with_suffix(".zip")
    curiosity = base.with_suffix(".intrinsic.pt")
    manifest = base.with_suffix(".pair.json")
    incomplete = base.with_suffix(".incomplete")
    if policy.exists() or curiosity.exists() or manifest.exists():
        raise FileExistsError(f"Refusing to overwrite an existing checkpoint pair: {base}")
    temporary = base.parent / (".checkpoint-" + uuid.uuid4().hex)
    staged_policy = temporary.with_suffix(".zip")
    staged_curiosity = temporary.with_suffix(".intrinsic.pt")
    staged_manifest = temporary.with_suffix(".json")
    incomplete.touch(exist_ok=False)
    try:
        model.save(str(staged_policy))
        torch.save(intrinsic, staged_curiosity)
        record = {"schema": 1, "timesteps": int(model.num_timesteps),
                  "policy_sha256": digest(staged_policy),
                  "intrinsic_sha256": digest(staged_curiosity)}
        staged_manifest.write_text(json.dumps(record, indent=2), encoding="utf-8")
        staged_curiosity.replace(curiosity)
        staged_policy.replace(policy)
        staged_manifest.replace(manifest)
        incomplete.unlink()
    finally:
        for path in (staged_policy, staged_curiosity, staged_manifest):
            path.unlink(missing_ok=True)


def verify_pair_manifest(policy):
    policy = Path(policy)
    if policy.with_suffix(".incomplete").exists():
        raise ValueError(f"Checkpoint publication was interrupted: {policy}")
    manifest = policy.with_suffix(".pair.json")
    if not manifest.exists():
        return None  # Legacy local pairs remain resumable after ZIP verification.
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != 1 or record.get("policy_sha256") != digest(policy)
            or record.get("intrinsic_sha256") != digest(policy.with_suffix(".intrinsic.pt"))):
        raise ValueError(f"Checkpoint pair integrity mismatch: {policy}")
    return record["timesteps"]
