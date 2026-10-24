# Project instructions

- Keep all repository code, documentation, configuration and generated labels in English.
- Preserve local ROMs, savestates, emulator profiles and checkpoints; never commit them.
- Use the project `.venv` and keep the existing MIT/third-party license boundaries.
- The owner requested stopping all training on October 23, 2026 before shutting down
  the PC. Online collection, offline fitting and policy training must remain stopped
  until the owner explicitly authorizes resuming. The local training hold is active.
- On October 24 the owner authorized bounded offline Galaxy 2 world-model ablations
  using copies of the retained checkpoint and existing replay. Only that experiment
  may fit models under its scoped local grant. Keep online collection and policy
  training stopped; preserve the original checkpoint and unchanged quality gates.
- Final world-model checkpoints are retained at 870272 Galaxy 2 and 278773 Galaxy 1
  real decisions. Preserve held-out validation gates and real/imagined accounting.
- Existing PPO final pairs are retained at 824416 Galaxy 2 and 220891 Galaxy 1 decisions.
- Run meaningful portable tests and Ruff before committing changes; never count detector
  fixtures, imagined trajectories or synthetic tests as trained mission completions.
