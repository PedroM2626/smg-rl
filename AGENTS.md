# Project instructions

- Keep all repository code, documentation, configuration and generated labels in English.
- Preserve local ROMs, savestates, emulator profiles and checkpoints; never commit them.
- Use the project `.venv` and keep the existing MIT/third-party license boundaries.
- The owner explicitly stopped all training while the world-model implementation is prepared.
  Do not start real-game collection, policy training or offline world-model fitting until the
  owner explicitly authorizes resuming. Synthetic software tests are allowed.
  The local `.local/training-hold.json` enforces this hold across training entry points.
  Do not remove it merely to make a command succeed. A later explicit resume instruction
  supersedes this operational hold; update this note when that happens.
- Existing PPO final pairs are retained at 824416 Galaxy 2 and 220891 Galaxy 1 decisions.
- Run meaningful portable tests and Ruff before committing changes; never count detector
  fixtures, imagined trajectories or synthetic tests as trained mission completions.
