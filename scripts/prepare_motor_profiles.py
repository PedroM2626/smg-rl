"""Create dated, separate motor-only and route-reward research configurations."""

from copy import deepcopy
from pathlib import Path

import yaml

from smg_rl.checkpoints import digest


def main():
    base = yaml.safe_load(Path('config/staged-smg1-control.yaml').read_text())
    base['actions'] = {'profile':'motor16-v1'}
    base['frontier'] = {'enabled':False}
    base['paths']['route'] = 'config/routes/smg1-initial-exit.json'
    base['paths']['dolphin'] = '.local/native-dolphin-frontier/Dolphin.exe'
    base['world_model'].update(disagreement_scale=0.0, entropy=0.0003, warmup_epsilon=0.15)
    base['training']['seed'] = 142
    for arm in ('control','route'):
        config = deepcopy(base)
        config['paths']['replay'] = f'runs/motor16-{arm}/replay'
        config['paths']['runs'] = f'runs/motor16-{arm}'
        config['paths']['user'] = f'.local/motor16-{arm}/template'
        if arm == 'route':
            config['rewards']['progress'] = 0.0
            config['rewards']['revision'] = 'grounded-initial-route-v1'
            config['route_reward'] = {'enabled':True, 'weight':5.0,
                                     'spec_sha256':digest('config/routes/smg1-initial-exit.json')}
        destination = Path(f'config/motor16-smg1-{arm}.yaml')
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')


if __name__ == '__main__':
    main()
