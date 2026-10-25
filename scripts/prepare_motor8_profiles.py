"""Create the owner's exact eight-action profiles for both supported games."""

from pathlib import Path

import yaml

from smg_rl.checkpoints import digest


def main():
    for game, source in [('smg1','config/staged-smg1-control.yaml'),
                          ('smg2','config/staged-smg2-terminal_reward_weight.yaml')]:
        config = yaml.safe_load(Path(source).read_text())
        config['actions'] = {'profile':'motor8-v1'}
        config['frontier'] = {'enabled':False}
        config['paths'].update(dolphin='.local/native-dolphin-frontier/Dolphin.exe',
                               replay=f'runs/m8-{game}/replay', runs=f'runs/m8-{game}',
                               user=f'.local/m8-{game}/template')
        config['world_model'].update(disagreement_scale=0.0, entropy=0.0003, warmup_epsilon=0.15)
        config['training']['seed'] = 142
        config['training']['steps'] = 256
        config['parallel']['workers'] = len(config['variations'])
        if game == 'smg1':
            config['paths']['route'] = 'config/routes/smg1-initial-exit.json'
            config['rewards'].update(progress=0.0, revision='grounded-initial-route-v1')
            config['route_reward'] = {'enabled':True, 'weight':5.0,
                                     'spec_sha256':digest('config/routes/smg1-initial-exit.json')}
        destination = Path(f'config/motor8-{game}.yaml')
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_text(yaml.safe_dump(config,sort_keys=False), encoding='utf-8')


if __name__ == '__main__':
    main()
