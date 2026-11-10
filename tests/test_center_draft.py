from pathlib import Path

from smg_rl.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_center_draft_cannot_be_trained_until_the_owner_approves_the_gate():
    draft = load_config(ROOT / 'config/center-bbf-draft-20261110.yaml')
    assert draft['training']['enabled'] is False
    assert draft['game']['stage'] == 'KoopaBattleVs3Galaxy'
    assert draft['regional_reward']['scale'] < draft['rewards']['completion']
    calibration = (ROOT / 'config/smg1-center-calibration.json').read_text(encoding='utf-8')
    assert '"validated": true' not in calibration
