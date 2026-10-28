"""Replay saved assisted discoveries and test their full-start lineage."""

import json
import os
from pathlib import Path

import numpy as np
import psutil

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.env import GalaxyEnv
from smg_rl.native import NativeDolphin
from smg_rl.parallel import runtime_directory
from smg_rl.policy_video import VideoBackend
from smg_rl.train import new_run

os.chdir(Path(__file__).resolve().parents[1])
SEARCH = Path('.local/chocolate-frontier-search-20261028/state.json')
ANCHOR = 'sweet-sweet-first-chocolate.state'
INITIAL_ACTIONS = [1] * 3 + [3] * 16 + [1] * 2 + [3] * 8 + [0] * 2


def matching(sample, position, scene_frame):
    return bool(not sample.dead and not sample.complete and sample.grounded is True and sample.health == 3
                and sample.scene_frame == scene_frame
                and np.linalg.norm(np.subtract(sample.position, position)) <= .01)


def candidates():
    state = json.loads(SEARCH.read_text(encoding='utf-8'))
    if state['phase'] != 'completed_needs_review' or len(state['workers']) != 6:
        raise ValueError('The finite search must complete before candidate selection.')
    values = []
    for worker in state['workers']:
        if worker['phase'] != 'completed' or worker['exit_code'] != 0:
            raise ValueError('Search worker needs diagnosis before selection.')
        for line in Path(worker['stdout']).read_text(encoding='utf-8').splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if 'run' not in value:
                continue
            report = json.loads((Path(value['run']) / 'summary.json').read_text(encoding='utf-8'))
            if report['state'] != 'completed' or len(report['trials']) != 24:
                raise ValueError('Search produced an incomplete worker budget.')
            for trial in report['trials']:
                if trial['candidate']:
                    entry = trial['candidate']
                    if digest(entry['file']) != entry['sha256']:
                        raise ValueError('Candidate snapshot integrity failed.')
                    values.append({'worker': worker['worker'], 'trial': trial['trial'], **entry})
            break
    # Deeper Z is a coarse search prioritization, not a mission progress estimate.
    return sorted(values, key=lambda item: item['position'][2])[:4]


def main():
    if any((p.info['name'] or '').lower() == 'dolphin.exe' for p in psutil.process_iter(['name'])):
        raise ValueError('Existing Dolphin processes require ownership review.')
    selected = candidates()
    if not selected:
        print(json.dumps({'state': 'no_candidates', 'selected': 0}))
        return
    config = load_config('config/smg1-six-two-hour.yaml')
    config['model_free_run'] = 'evaluation'
    config['replay']['enabled'] = False
    config['paths']['runs'] = str(Path(config['root']) / 'runs/chocolate-frontier-validation-20261028')
    run = new_run(config, 'validate')
    config['paths']['user'] = str(runtime_directory(config, run) / 'template')
    native_config = dict(config)
    native_config['parallel'] = dict(config['parallel'], frame_skip=2, native_capture='full')
    backend = NativeDolphin(native_config)
    video = VideoBackend(backend, run / 'validation.mp4', 'SCRIPTED ASSISTED AND FULL-START VALIDATION',
                         decision_frames=12)
    env = None
    result = {'state': 'running', 'run': str(run), 'candidates': [], 'physical_decisions': 0,
              'scope': 'Scripted validation; full-start path evidence only, never learned success.'}
    atomic_json(run / 'result.json', result)
    print(json.dumps({'run': str(run), 'selected': len(selected)}), flush=True)
    try:
        env = GalaxyEnv(config, video, run / 'worker-00', memory=backend.memory, intrinsic=False, variation=0)
        reset_seed = next(i for i in range(10000)
                          if (entry := env.curriculum_bank.choose(np.random.default_rng(i), 1))
                          and entry['file'] == ANCHOR)

        def play(actions, observation):
            info = {}
            for action in actions:
                video.policy_image = observation['image'][-1].copy()
                video.policy_action_source = 'scripted_fixture'
                video.policy_q_values = None
                observation, _, terminal, truncated, info = env.step(action)
                result['physical_decisions'] += 1
                if terminal or truncated:
                    return observation, info, False
            return observation, info, True

        for candidate in selected:
            item = {'candidate': candidate, 'assisted_replays': [], 'timing_perturbation': None,
                    'full_start_replay': None, 'accepted': False}
            result['candidates'].append(item)
            for repetition in range(2):
                observation, _ = env.reset(seed=reset_seed, options={'variation': 0})
                if env.start_kind != 'curriculum' or env.start_state_id != ANCHOR:
                    raise ValueError('Assisted replay restored the wrong anchor.')
                video.episode = len(result['candidates']) * 10 + repetition
                observation, _, completed = play(candidate['actions'], observation)
                same = completed and matching(env.telemetry, candidate['position'], candidate['scene_frame'])
                start = np.array(env.telemetry.position)
                alive = bool(same)
                displacement = None
                if same:
                    for _ in range(5):
                        observation, _, alive_step = play([0], observation)
                        alive = bool(alive and alive_step and not env.telemetry.dead
                                     and env.telemetry.grounded is True and env.telemetry.health == 3)
                        if not alive:
                            break
                    displacement = float(np.linalg.norm(np.subtract(env.telemetry.position, start)))
                item['assisted_replays'].append({'exact_candidate': same, 'idle_alive_grounded_60_frames': alive,
                                                 'idle_displacement_xyz': displacement})
            observation, initial = env.reset(seed=reset_seed, options={'variation': 0, 'full_start': True})
            video.episode = len(result['candidates']) * 10 + 2
            observation, _, anchor_reached = play(INITIAL_ACTIONS, observation)
            anchor_entry = next(entry for entry in env.curriculum_bank.entries if entry['file'] == ANCHOR)
            anchor_match = anchor_reached and matching(env.telemetry, anchor_entry['position'], anchor_entry['scene_frame'])
            candidate_match = False
            if anchor_match:
                observation, _, reached = play(candidate['actions'], observation)
                candidate_match = reached and matching(env.telemetry, candidate['position'], candidate['scene_frame'])
            item['full_start_replay'] = {'initial': initial, 'anchor_exact': anchor_match,
                                         'candidate_exact': candidate_match, 'final': env.telemetry.to_dict()}
            observation, _ = env.reset(seed=reset_seed, options={'variation': 0})
            video.episode = len(result['candidates']) * 10 + 3
            observation, _, delayed = play([0, 0], observation)
            if delayed:
                observation, _, delayed = play(candidate['actions'], observation)
            item['timing_perturbation'] = {'two_noop_decisions_before_actions': True, 'completed': delayed,
                                           'final': env.telemetry.to_dict()}
            item['accepted'] = bool(candidate_match and all(r['exact_candidate'] and r['idle_alive_grounded_60_frames']
                and r['idle_displacement_xyz'] <= 250 for r in item['assisted_replays']))
            atomic_json(run / 'result.json', result)
        result['state'] = 'completed'
    except BaseException as exc:
        result.update(state='failed', error=repr(exc))
        raise
    finally:
        (env or video).close()
        result['frames_written'] = video.frames_written
        atomic_json(run / 'result.json', result)
    print(json.dumps({'state': result['state'], 'run': str(run),
                      'accepted': sum(c['accepted'] for c in result['candidates'])}), flush=True)


if __name__ == '__main__':
    main()
