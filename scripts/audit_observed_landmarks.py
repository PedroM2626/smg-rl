"""Audit human landmark candidates against a replayed completion and frozen trials."""

import argparse
import csv
import json
from pathlib import Path

import cv2

from smg_rl.checkpoints import digest
from smg_rl.human_route_audit import (
    episode_landmark_candidates,
    load_spec,
    observed_landmark_candidates,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demonstration', required=True, type=Path)
    parser.add_argument('--replay-result', required=True, type=Path)
    parser.add_argument('--evaluation-report', required=True, type=Path)
    parser.add_argument('--landmarks', default='config/evaluation/smg1-human-landmarks-v1.json', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('A new audit must not overwrite prior evidence.')
    demo = args.demonstration.resolve()
    manifest = json.loads((demo / 'manifest.json').read_text(encoding='utf-8'))
    replay = json.loads(args.replay_result.read_text(encoding='utf-8'))
    specification = load_spec(args.landmarks)
    if (manifest['state'] != 'completed' or manifest['game_id'] != specification['game_id']
            or manifest['scenario'] != specification['scenario'] or manifest['start_slot'] != 1
            or not manifest['full_start_ram_star'] or replay['state'] != 'star'
            or Path(replay['source']).resolve() != demo or not replay['ram_star']
            or replay['real_replay_samples'] != manifest['samples']
            or replay['first_position_divergence'] is not None
            or replay['first_observation_divergence'] is not None
            or not replay['source_savestate_unchanged']
            or manifest['source_state_sha256'] != specification['source_state_sha256']):
        raise ValueError('A matching exactly reproduced canonical human star is required.')
    for name, expected in (('trajectory.csv', specification['human_trajectory_sha256']),
                           ('gameplay.mp4', specification['human_video_sha256'])):
        if manifest['files_sha256'][name] != expected or digest(demo / name) != expected:
            raise ValueError('Human reference artifact failed integrity verification.')
    with (demo / 'trajectory.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != manifest['samples'] + 1:
        raise ValueError('Human input and trajectory counts differ.')
    video = cv2.VideoCapture(str(demo / 'gameplay.mp4'))
    try:
        if not video.isOpened():
            raise ValueError('The human source video cannot be decoded.')
        for target in specification['landmarks']:
            index = target['human_sample']
            if (target['video_frame'] != index - 1 or not 0 < index < len(rows) - 1
                    or rows[index]['grounded'] != 'True'
                    or [float(rows[index][k]) for k in ('x', 'y', 'z')] != target['position']):
                raise ValueError('A landmark does not match a real grounded source observation.')
            video.set(cv2.CAP_PROP_POS_FRAMES, target['video_frame'])
            if not video.read()[0]:
                raise ValueError('A landmark source video frame cannot be decoded.')
    finally:
        video.release()
    human_phases = []
    for phase in range(6):
        sampled = [rows[0]] + [rows[i] for i in range(1 + phase, len(rows), 6)]
        matches = observed_landmark_candidates(sampled, specification)
        human_phases.append({'phase': phase, 'matches': matches})
        if len(matches) != len(specification['landmarks']):
            raise ValueError('At least one 12-frame phase cannot find all reference landmarks.')
    report = json.loads(args.evaluation_report.read_text(encoding='utf-8'))
    results = []
    for arm, data in report['arms'].items():
        for evaluation in data['evaluations']:
            for ep in evaluation['episodes']:
                path = Path(ep['trajectory'])
                if digest(path) != ep['trajectory_sha256']:
                    raise ValueError('Frozen episode trajectory changed.')
                result = episode_landmark_candidates(path, specification)
                if result['native_star'] != ep['native_star']:
                    raise ValueError('Frozen episode native star evidence disagrees.')
                results.append({'arm': arm, 'boundary': evaluation['boundary'],
                    'mode': evaluation['mode'], 'seed': ep['seed'], 'outcome': ep['outcome'],
                    'trajectory': str(path), 'trajectory_sha256': ep['trajectory_sha256'], **result})
    evidence = {'state': 'completed', 'landmark_spec_sha256': digest(args.landmarks),
                'source_replay_sha256': digest(args.replay_result),
                'evaluation_report_sha256': digest(args.evaluation_report),
                'human_sampled_phases': human_phases, 'evaluation_episodes': results,
                'limitations': ['Human trajectory reused for its own landmark construction; not independent validation.',
                    'Position proximity creates only candidate visits; visual review is necessary, especially on moving platforms.',
                    'Alternative valid routes and between-sample crossings may be missed.',
                    'Landmark count is not level completion or a native percentage.',
                    'The previously reported wide-corridor projection is withdrawn as progress.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'human_12_frame_phases': len(human_phases), 'human_landmarks': len(specification['landmarks']),
                      'evaluation_episodes': len(results),
                      'evaluation_landmark_candidates': sum(len(x['candidate_landmarks']) for x in results),
                      'evaluation_stars': sum(x['native_star'] for x in results)}))


if __name__ == '__main__':
    main()
