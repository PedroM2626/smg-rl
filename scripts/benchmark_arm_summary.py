"""Per-block progress table of an algorithm-benchmark arm, read from its attempts ledgers.

Usage: python scripts/benchmark_arm_summary.py runs/model-free/algorithm-benchmark-20261106/F
Counts only policy attempts (never scripted demonstrations). `opening_advance` is the fraction
of the route covered, a progress measure and not a mission percentage.
"""
import json
import statistics
import sys
from pathlib import Path


def summarize(root):
    rows = []
    for run in sorted(Path(root).glob('dqn-*')):
        ledger, status = run / 'attempts.jsonl', run / 'status.json'
        if not ledger.exists() or not status.exists():
            continue
        attempts = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        policy = [a for a in attempts if a['start_kind'] != 'scripted_demonstration']
        if not policy:
            continue
        recent = policy[-100:]
        rows.append({
            'run': run.name, 'state': json.loads(status.read_text())['state'],
            'real_decisions': attempts[-1]['real_decisions'], 'attempts': len(policy),
            'stars': sum(bool(a['ram_star']) for a in policy),
            'deaths': sum(bool(a['dead']) for a in policy),
            'best_advance': max(a['opening_advance'] for a in policy),
            'recent_mean_advance': statistics.fmean(a['opening_advance'] for a in recent),
            'recent_median_decisions': statistics.median(a['decisions'] for a in recent)})
    return rows


if __name__ == '__main__':
    for row in summarize(sys.argv[1]):
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
