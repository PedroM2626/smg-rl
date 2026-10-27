"""Keep the primary documentation tied to checked evidence and local links."""

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_local_markdown_links_resolve():
    files = [ROOT / 'README.md', ROOT / 'SECURITY.md', ROOT / 'data' / 'README.md']
    files.extend((ROOT / 'docs').rglob('*.md'))
    missing = []
    for path in files:
        text = path.read_text(encoding='utf-8')
        for match in re.finditer(r'!?\[[^\]]+\]\(([^)]+)\)', text):
            link = match.group(1).split()[0].strip('<>')
            parsed = urlsplit(link)
            if parsed.scheme or not parsed.path or parsed.path.startswith('/'):
                continue
            if not (path.parent / unquote(parsed.path)).exists():
                missing.append((path.relative_to(ROOT), link))
    assert not missing, missing


def test_latest_iqn_documentation_matches_recorded_evidence_and_profile():
    result = json.loads((ROOT / 'docs/assets/performance/iqn-four-hour-result-20261027.json')
                        .read_text(encoding='utf-8'))
    profile = yaml.safe_load((ROOT / 'config/smg1-six-two-hour.yaml').read_text(encoding='utf-8'))
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    status = (ROOT / 'docs/current-status.md').read_text(encoding='utf-8')
    assert result['training_completed_four_hours'] is False
    assert result['last_valid_checkpoint_real'] == 749006
    assert result['last_valid_checkpoint_completed_attempts'] == 12125
    assert result['uncheckpointed_acknowledged_decisions'] == 97668
    assert f"{result['last_valid_checkpoint_real']:,}" in readme
    assert f"{result['last_valid_checkpoint_real']:,}" in status
    assert profile['dqn']['algorithm'] == 'iqn'
    assert profile['dqn']['encoder'] == 'impala'
    assert profile['actions']['profile'] == 'motor8-v1'
    assert profile['regional_reward']['mode'] == 'advance'
    assert profile['rewards']['revision'] == 'regional-advance-undiscounted-v2'
    assert profile['rewards']['intrinsic'] == 0
