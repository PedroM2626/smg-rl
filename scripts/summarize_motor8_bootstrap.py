"""Summarize only completed motor8 real collection up to the 4096 boundary."""

import csv
import json
from collections import Counter
from pathlib import Path

from smg_rl.diagnostics import atomic_json
from smg_rl.plotting import plot_trajectory


def main():
    state=json.loads(Path('.local/motor8-state.json').read_text())
    output=Path('docs/assets/performance/motor8-bootstrap-review-20261025.json')
    result=json.loads(output.read_text())
    result['per_mission_collection']={}
    for job in state['jobs']:
        missions={}
        for run_name in job['runs']:
            run=Path(run_name)
            if not run.name.startswith('world-model-2'):
                continue
            status=json.loads((run/'status.json').read_text())
            if status['state']!='completed' or status['real_decisions']>4096:
                continue
            for path in run.glob('worker-*/*/episode-*.csv'):
                with path.open(newline='',encoding='utf-8') as stream:
                    rows=list(csv.DictReader(stream))
                if not rows or rows[0].get('start_kind')!='full_start':
                    raise ValueError('Bootstrap summary requires recorded full-start real episodes.')
                name=rows[0]['variation']
                summary=missions.setdefault(name,{'real_decisions':0,'episodes':0,'outcomes':Counter(),
                    'ram_mission_completions':0,'hp_lost':0 if job['game']=='smg1' else None,
                    'maximum_geometric_projection':0.0,'maximum_initial_prefix_fraction':None,
                    'best_geometric_csv':None,'confirmed_initial_exit_episodes':None})
                summary['episodes']+=1
                summary['real_decisions']+=len(rows)-1
                summary['outcomes'][rows[-1]['outcome']]+=1
                summary['ram_mission_completions']+=int(rows[-1]['complete']=='True')
                if summary['hp_lost'] is not None:
                    summary['hp_lost']+=sum(int(row.get('health_lost') or 0) for row in rows)
                projection=max(float(row['best_progress']) for row in rows)
                if summary['best_geometric_csv'] is None or projection>summary['maximum_geometric_projection']:
                    summary['maximum_geometric_projection']=projection
                    summary['best_geometric_csv']=str(path)
                prefixes=[float(row['route_prefix_progress']) for row in rows
                          if row.get('route_prefix_progress') not in (None,'')]
                if prefixes:
                    summary['maximum_initial_prefix_fraction']=max(summary['maximum_initial_prefix_fraction'] or 0,max(prefixes))
                    summary['confirmed_initial_exit_episodes']=(summary['confirmed_initial_exit_episodes'] or 0)+int(
                        any(row.get('route_prefix_exited')=='True' for row in rows))
        if sum(m['real_decisions'] for m in missions.values())!=4096:
            raise ValueError('Both games must finish their exact 4096-decision bootstrap before summarizing.')
        for name, summary in missions.items():
            summary['interpretation']='Fixed-prior real collection, no actor/model updates during bootstrap. Projection is geometric, not native mission or verified traversal percentage.'
            if job['game']=='smg2':
                graph=f'docs/assets/performance/motor8-smg2-{name}-bootstrap-20261025.png'
                plot_trajectory(summary['best_geometric_csv'],graph)
                summary['trajectory']=graph
        result['per_mission_collection'][job['game']]=missions
    atomic_json(output,result)
    print(json.dumps(result['per_mission_collection'],indent=2))


if __name__=='__main__':
    main()
