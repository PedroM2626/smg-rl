"""Read-only paired audit on reused held-out episodes before further motor8 work."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from smg_rl.checkpoints import digest
from smg_rl.config import load_config
from smg_rl.diagnostics import atomic_json
from smg_rl.replay import SequenceReplay, replay_signature
from smg_rl.world_ablation import METRICS, evaluate, freeze_validation
from smg_rl.world_model import ImaginedAgent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game',choices=['smg1','smg2'],required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Retain previous audits.')
    torch.set_num_threads(1)
    config=load_config(f'config/motor8-{args.game}.yaml')
    signature=replay_signature(config)
    replay=SequenceReplay(config['paths']['replay'],signature)
    summary=replay.refresh()
    if summary['rejected']:
        raise ValueError('Rejected replay prevents an audit.')
    settings=json.loads(Path(args.baseline).with_suffix('.json').read_text())['settings']
    fixed,episodes=freeze_validation(replay,settings,8,True)
    result={'scope':'Paired read-only comparison on reused held-out episodes; no new real or imagined training, optimizer or counter updates. Seeds are validation draws, not independent training replications.',
        'game':args.game,'seeds':[11,23,47],'audit_episodes':episodes,'replay':summary,'rows':[]}
    for name,checkpoint in [('bootstrap',args.baseline),('fitted',args.candidate)]:
        agent=ImaginedAgent.load(checkpoint,signature,settings,config['training']['device'])
        before=(agent.real_decisions,agent.model_updates,agent.actor_updates,agent.imagined_transitions)
        row={'name':name,'checkpoint':checkpoint,'sha256':digest(checkpoint),
             'counters':dict(zip(('real_decisions','model_updates','actor_updates','imagined_transitions'),before)),
             'audits':[]}
        for seed in result['seeds']:
            row['audits'].append({'seed':seed,**evaluate(agent,fixed,settings,seed,summary)})
        assert before==(agent.real_decisions,agent.model_updates,agent.actor_updates,agent.imagined_transitions)
        assert digest(checkpoint)==row['sha256']
        row['gate_passes']=sum(a['gate']['ready'] for a in row['audits'])
        row['mean_metrics']={scenario:{metric:float(np.mean([a['metrics'][scenario][metric] for a in row['audits']]))
                                    for metric in METRICS} for scenario in signature['scenario_names']}
        result['rows'].append(row)
        atomic_json(output,result)
        print(json.dumps({'name':name,'gate_passes':row['gate_passes'],'mean_metrics':row['mean_metrics']}),flush=True)
        del agent
    result['state']='completed'
    atomic_json(output,result)


if __name__=='__main__':
    main()
