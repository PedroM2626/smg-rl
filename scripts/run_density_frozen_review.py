"""Record finite frozen evaluations of the two audited SMG1 density checkpoints."""

import ast
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.dqn_trial import audit_block
from smg_rl.human_route_audit import episode_landmark_candidates, load_spec
from smg_rl.resource_disk import require_disk_headroom
from smg_rl.resource_memory import memory_snapshot

ROOT = Path('.local/density-frozen-review-20261029')
STATE = ROOT / 'state.json'
PREVIOUS = Path('.local/human-route-density-recovery-20261028/state.json')
LANDMARKS = Path('config/evaluation/smg1-human-landmarks-v1.json')


def save(state):
    if not atomic_json(STATE, state):
        raise OSError('Evaluation ledger could not be persisted.')


def parse_run(path):
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try:
            item = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(item, dict) and 'evaluation_run' in item:
            return Path(item['evaluation_run'])
    return None


def stop_owned_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=10)
        for process in alive:
            process.kill()
    except psutil.NoSuchProcess:
        pass


def main():
    if ROOT.exists():
        raise FileExistsError('A frozen review exists; never duplicate it.')
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise RuntimeError('Existing Dolphin instances require ownership review.')
    source = json.loads(PREVIOUS.read_text(encoding='utf-8'))
    if source['phase'] != 'needs_review' or source.get('error') != "RuntimeError('Recovery stopped for review: memory_pressure')":
        raise ValueError('Expected the previously audited resource-stopped recovery.')
    last = source['jobs'][-1]
    partial = audit_block(Path(last['run']), load_config(last['config']), last['source_real'],
                          last['source_attempts'], resource_stopped=True)
    arms = source['arms']
    arms['control']['checkpoint'] = partial['checkpoint']
    state = {'phase': 'prepared', 'source_state': str(PREVIOUS), 'source_state_sha256': digest(PREVIOUS),
             'partial_control_review': partial, 'arms': {}, 'jobs': [],
             'scope': 'Frozen policy evaluation; no training or imagined transitions; shared Wii reset state.'}
    for name, arm in arms.items():
        checkpoint = Path(arm['checkpoint'])
        manifest = json.loads((checkpoint / 'manifest.json').read_text(encoding='utf-8'))
        if (digest(checkpoint / 'agent.pt') != manifest['agent_sha256']
                or digest(checkpoint / 'replay.npz') != manifest['replay_sha256']):
            raise ValueError('Source agent or replay failed its integrity check.')
        config = load_config(arm['config'])
        require_disk_headroom(config)
        state['arms'][name] = {'config': arm['config'], 'config_sha256': digest(Path(arm['config'])),
                               'checkpoint': str(checkpoint), 'agent_sha256': manifest['agent_sha256'],
                               'replay_sha256': manifest['replay_sha256'], 'source_real': manifest['real_decisions']}
    ROOT.mkdir(parents=True)
    save(state)
    state.update(phase='running', supervisor={'pid': os.getpid(), 'created_time': psutil.Process().create_time()},
                 started_utc=datetime.now(timezone.utc).isoformat())
    save(state)
    try:
        for arm_name in ('candidate', 'control'):
            arm = state['arms'][arm_name]
            for mode in ('greedy', 'stochastic'):
                if memory_snapshot()['physical_available_bytes'] < 6 * 2**30:
                    raise MemoryError('Insufficient physical memory for evaluation startup.')
                if digest(Path(arm['config'])) != arm['config_sha256']:
                    raise ValueError('Evaluation config changed.')
                label = f'{arm_name}-{mode}'
                job = {'label': label, 'arm': arm_name, 'mode': mode, 'phase': 'pending',
                       'stdout': str(ROOT / f'{label}.stdout.log'), 'stderr': str(ROOT / f'{label}.stderr.log')}
                job['command'] = [sys.executable, '-m', 'smg_rl.cli', '--config', arm['config'], 'record-dqn',
                                  arm['checkpoint'], '--variation', '0', '--max-steps', '600', '--seeds']
                job['command'] += ['11'] if mode == 'greedy' else ['11', '23', '47', '--stochastic', '--evaluation-epsilon', '0.1']
                state['jobs'].append(job)
                save(state)
                env = dict(os.environ)
                env.update({k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
                with open(job['stdout'], 'x', encoding='utf-8') as out, open(job['stderr'], 'x', encoding='utf-8') as err:
                    process = subprocess.Popen(job['command'], stdout=out, stderr=err, env=env,
                                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                job.update(phase='running', pid=process.pid, created_time=psutil.Process(process.pid).create_time())
                save(state)
                end = time.monotonic() + 1200
                while process.poll() is None:
                    run = parse_run(job['stdout'])
                    if run and job.get('run') != str(run):
                        job['run'] = str(run)
                        save(state)
                    if time.monotonic() >= end or memory_snapshot()['physical_available_bytes'] < 2 * 2**30:
                        stop_owned_tree(process.pid)
                        raise RuntimeError('Evaluation exceeded time or memory safety bound; preserve partial files.')
                    time.sleep(3)
                job['exit_code'] = process.returncode
                save(state)
                if process.returncode:
                    raise RuntimeError('Evaluation failed; preserve partial files without retry.')
                run = parse_run(job['stdout'])
                if run is None:
                    raise ValueError('Evaluation did not publish its run.')
                record = json.loads((run / 'recording.json').read_text(encoding='utf-8'))
                expected_count = 1 if mode == 'greedy' else 3
                if (record['state'] != 'completed' or len(record['episodes']) != expected_count
                        or record['agent_sha256'] != arm['agent_sha256']
                        or digest(Path(arm['checkpoint']) / 'agent.pt') != arm['agent_sha256']
                        or digest(Path(arm['checkpoint']) / 'replay.npz') != arm['replay_sha256']):
                    raise ValueError('Evaluation identity, episode count or frozen checkpoint changed.')
                spec = load_spec(LANDMARKS)
                episodes = []
                for index, episode in enumerate(record['episodes'], 1):
                    trajectory = run / 'worker-00' / 'sweet_sweet' / f'episode-{index:06d}.csv'
                    try:
                        observed = episode_landmark_candidates(trajectory, spec)
                    except ValueError as exc:
                        observed = {'excluded': str(exc)}
                    else:
                        if observed['native_star'] != episode['full_start_ram_star']:
                            raise ValueError('Native completion evidence disagrees.')
                    episodes.append({'seed': episode['seed'], 'decisions': episode['decisions'],
                        'outcome': episode['outcome'], 'native_star': episode['full_start_ram_star'],
                        'epsilon_random_decisions': episode['epsilon_random_decisions'],
                        'trajectory': str(trajectory), 'trajectory_sha256': digest(trajectory),
                        'observation': observed})
                job.update(phase='reviewed', run=str(run), episodes=episodes, video=record['video'],
                           agent_sha256=record['agent_sha256'], optimizer_updates=record['optimizer_updates'])
                save(state)
        state['phase'] = 'completed_needs_scientific_review'
    except BaseException as exc:
        state.update(phase='needs_review', error=repr(exc))
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(state)


if __name__ == '__main__':
    main()
