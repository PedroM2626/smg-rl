# Vendored Beyond The Rainbow (BTR) agent code

Files here are copied unchanged from https://github.com/VIPTankz/BTR at commit
`093ca77f08540d646f12fd942ee5fda763e0fd8a` (`Agent.py`, `PER.py`, `networks.py`, `Analytic.py`),
under the MIT License of that repository (see `LICENSE`, copyright 2024 VIPTankz). They are the code of
Clark, Towers, Evers and Hare, "Beyond The Rainbow: High Performance Deep Reinforcement Learning on a
Desktop PC", ICML 2025. Nothing in this directory was written by this project.

This project only imports them through `src/smg_rl/official_btr.py`, which supplies a stand-in for the
unused `torchvision` import of `Agent.py` and drives the agent with this project's environment workers. The
VIPTankz/Wii-RL repository has no licence file and none of its code is used here.
