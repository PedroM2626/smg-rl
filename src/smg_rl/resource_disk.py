"""Keep checkpoint space on every owned output/profile volume before collection."""

import shutil
from pathlib import Path


def disk_snapshot(config):
    result = {}
    for key in ('runs', 'user'):
        path = Path(config['paths'][key]).resolve()
        while not path.exists() and path != path.parent:
            path = path.parent
        usage = shutil.disk_usage(path)
        result[key] = {'path': str(path), 'free_bytes': usage.free, 'total_bytes': usage.total}
    return result


def disk_safe(snapshot, reserve_bytes=4 * 2**30):
    return all(item['free_bytes'] >= reserve_bytes for item in snapshot.values())


def require_disk_headroom(config):
    snapshot = disk_snapshot(config)
    if not disk_safe(snapshot):
        raise OSError(f'Insufficient disk headroom for collection and final checkpoint: {snapshot}')
    return snapshot
