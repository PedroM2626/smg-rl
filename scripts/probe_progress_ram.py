"""Read-only SMG1 initial-prefix RAM probe, not a validated progress detector."""

import json

import numpy as np

from smg_rl.actions import action_set
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.memory_probe import candidates
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.train import new_run


def main():
    config = load_config('config/motor8-smg1.yaml')
    config['replay']['enabled'] = False
    config['frontier'] = {'enabled': False}
    config['paths']['runs'] = 'runs/m8-smg1-boundary'
    run = new_run(config, 'ram-progress-probe')
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    result = {'scope': 'Read-only aligned MEM1 scalar screening during a scripted initial-prefix '
              'fixture. Not trained navigation, native progress validation or a memory write. '
              'Only small nonnegative integers and bounded float32 values screened; no claim '
              'about other encodings, MEM2, later route segments or other games.',
              'maximum_physical_decisions': 33, 'physical_decisions': 0, 'samples': [],
              'candidate_addresses_are_unvalidated': True, 'candidates': []}
    backend = NativeDolphin(config)
    env = GalaxyEnv(config, backend, run / 'worker-00', backend.memory, intrinsic=False)
    files = []

    def snapshot(label):
        data = backend.memory.reader.read_bytes(0x80000000, 0x1800000)
        path = run / f'mem1-{len(files)}.bin'
        path.write_bytes(data)
        files.append(path)
        result['samples'].append({'label': label, 'telemetry': backend.memory.sample().to_dict(),
                                  'file': str(path)})

    def advance(names, count):
        for _ in range(count):
            _, _, terminated, truncated, _ = env.step(action_set(config).index(names))
            result['physical_decisions'] += 1
            if terminated or truncated:
                raise RuntimeError('The probe left its validated live prefix; retain partial evidence.')

    try:
        env.reset(seed=11)
        snapshot('initial')
        advance((), 10)
        snapshot('initial_stationary_after120_frames')
        env.reset(seed=11)
        snapshot('initial_reset')
        advance(('d',), 3)
        snapshot('right36_frames')
        for step in range(4):
            advance(('w',), 4)
            snapshot(f'forward{48 * (step + 1)}_frames')
        advance(('s',), 4)
        snapshot('backward48_frames')
        env.reset(seed=11)
        snapshot('final_reset')
    finally:
        atomic_json(run / 'probe.json', result)
        env.close()
    positions = [s['telemetry']['position'] for s in result['samples']]
    for offset in range(0, 0x1800000, 0x100000):
        blobs = []
        for path in files:
            with path.open('rb') as stream:
                stream.seek(offset)
                blobs.append(stream.read(0x100000))
        for kind in ('>f4', '>u4'):
            values = np.stack([np.frombuffer(data, dtype=kind).astype(np.float64) for data in blobs])
            indices = candidates(values, positions)
            for index in indices:
                if len(result['candidates']) < 128:
                    result['candidates'].append({'address': f'0x{0x80000000 + offset + 4 * index:08X}',
                                                 'kind': kind, 'values': values[:, index].tolist()})
            result.setdefault('candidate_counts', {}).setdefault(kind, 0)
            result['candidate_counts'][kind] += len(indices)
    result['state'] = 'completed_screening_only'
    result['interpretation'] = 'Every surviving value is only a correlation candidate. '
    result['interpretation'] += 'Require stationary/jump/spin/crouch, reverse/revisit, fall/death, '
    result['interpretation'] += 'multiple segment/region crossings, reset/pointer stability and end-state tests before accepting a reward signal.'
    atomic_json(run / 'probe.json', result)
    print(json.dumps({'run': str(run), 'physical_decisions': result['physical_decisions'],
                      'candidate_counts': result['candidate_counts'],
                      'accepted_native_progress_addresses': 0}), flush=True)


if __name__ == '__main__':
    main()
