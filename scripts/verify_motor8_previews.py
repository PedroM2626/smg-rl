"""Verify generated movie streams and exact-input sidecars; retain friendly copies."""

import json
import shutil
import subprocess
from pathlib import Path

from smg_rl.checkpoints import digest
from smg_rl.diagnostics import atomic_json


def main():
    destination = Path('runs/motor8-previews-20261025')
    destination.mkdir(exist_ok=True)
    rows = []
    for game in ('smg1','smg2'):
        for manifest in sorted(Path(f'runs/m8-{game}').glob('eval-world-video-*/recording.json')):
            data = json.loads(manifest.read_text())
            if data.get('source_actor_updates') != 0 or data.get('seeds') != [11]:
                continue  # Later trained evaluations must never be labelled as prior previews.
            movie = Path(data['video'])
            trace = [json.loads(line) for line in movie.with_suffix('.inputs.jsonl').read_text().splitlines()]
            assert len(trace) == data['frames_written']
            assert all(len(row['applied_controls']) <= 1 for row in trace)
            groups = {}
            for row in trace:
                groups.setdefault((row['episode'],row['decision']),[]).append(row)
            for (_,decision), samples in groups.items():
                assert len({r['policy_input_sha256'] for r in samples}) == 1
                if decision:
                    assert len(samples) == 6
                    if samples[0]['selected_action'] in ('jump','spin'):
                        assert all(r['applied_controls'] == [samples[0]['selected_action']] for r in samples[:4])
                        assert all(r['applied_controls'] == [] for r in samples[4:])
            probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,codec_name,avg_frame_rate,nb_frames','-show_entries','format=duration',
                '-of','json',str(movie)],text=True))
            assert int(probe['streams'][0]['nb_frames']) == len(trace)
            output = destination / f"{game}-{data['variation']}-prior.mp4"
            if output.exists():
                assert digest(output) == digest(movie)
            else:
                shutil.copy2(movie,output)
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','2','-i',str(movie),
                            '-frames:v','1',str(destination/f"{game}-{data['variation']}-preview.png")],check=True)
            rows.append({'game':game,'variation':data['variation'],'run':str(manifest.parent),
                'video':str(output),'sha256':digest(output),'source_checkpoint_sha256':data['checkpoint_sha256'],
                'source_actor_updates':data['source_actor_updates'],'episodes':data['episodes'],
                'frames':len(trace),'probe':probe,'sidecar_checks':'Exclusive controls; stable input digest per decision; correct8-on/4-off jump/spin samples'})
    assert len(rows) == 3
    atomic_json('docs/assets/performance/motor8-video-previews-20261025.json',{
        'scope':'Untrained motor8 policy-prior previews, no learning or trained mission-success evidence',
        'rows':rows})
    print(json.dumps({'verified_videos':len(rows),'destination':str(destination)}))


if __name__ == '__main__':
    main()
