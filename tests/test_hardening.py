import json
import struct
import zipfile
import zlib

import pytest

from smg_rl.checkpoints import save_pair
from smg_rl.dsu import valid_request
from smg_rl.scheduling import parse_cpu_sets, performance_cpus
from smg_rl.supervision import checkpoint_steps, newest_pair


class Model:
    num_timesteps = 42

    def save(self, path):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", json.dumps({"num_timesteps": self.num_timesteps}))


def test_checkpoint_publication_and_corruption_detection(tmp_path):
    base = tmp_path / "model-42"
    save_pair(Model(), {"test": 42}, base)
    assert checkpoint_steps(base.with_suffix(".zip")) == 42
    with pytest.raises(FileExistsError):
        save_pair(Model(), {}, base)
    base.with_suffix(".intrinsic.pt").write_bytes(b"damaged")
    assert newest_pair(tmp_path) == (0, None)


def test_interrupted_pair_never_selected(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("Simulated disk failure")

    monkeypatch.setattr("torch.save", fail)
    base = tmp_path / "model-42"
    with pytest.raises(OSError):
        save_pair(Model(), {}, base)
    assert base.with_suffix(".incomplete").exists()
    assert not list(tmp_path.glob(".checkpoint-*"))
    assert newest_pair(tmp_path) == (0, None)


def test_cpu_topology_variable_records_and_affinity():
    record = bytearray(32)
    struct.pack_into("<IIIH6B", record, 0, 32, 0, 1, 0, 2, 1, 0, 0, 3, 0)
    parsed = parse_cpu_sets(bytes(record) + struct.pack("<II", 8, 99))
    assert parsed[0]["logical"] == 2 and parsed[0]["efficiency"] == 3
    slower = dict(parsed[0], logical=4, efficiency=1)
    assert performance_cpus([*parsed, slower], [2, 4]) == [2]
    assert performance_cpus([*parsed, slower], [4]) == [4]
    with pytest.raises(ValueError):
        parse_cpu_sets(bytes(8))


def test_dsu_rejects_bad_lengths_and_checksums():
    request = bytearray(struct.pack("<4sHHIII", b"DSUC", 1001, 4, 0, 123, 0x100000))
    struct.pack_into("<I", request, 8, zlib.crc32(request))
    assert valid_request(request)
    assert not valid_request(request + b"unexpected")
    request[19] ^= 1
    assert not valid_request(request)
    assert not valid_request(b"DSUC")
