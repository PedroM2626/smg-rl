# License scope and third-party material

The original SMG RL source code, tests, configuration, technical documentation,
and project-authored figures are licensed under the [MIT License](LICENSE).
The Python distribution declares the SPDX license expression `MIT` and includes
the license and this notice in its distribution metadata.

## Game content and screenshots

Super Mario Galaxy, Super Mario Galaxy 2, their characters, game graphics,
audio, and other game content belong to their respective rights holders,
including Nintendo. The gameplay screenshot
[`docs/assets/smg1/damage-health2.png`](docs/assets/smg1/damage-health2.png)
documents a measured health transition. The MIT grant does not relicense the
underlying game artwork in that screenshot. Project-generated trajectory
plots and numeric telemetry fixtures are distinct from game artwork.

ROMs and savestates are excluded from Git and the Python distribution. Game
images are supplied by the user. Savestate bundles (RAM snapshots of the games)
are published separately as release assets for research reproducibility; this
repository grants no rights to the games or to content derived from them.

## Emulator, research sources, and dependencies

Dolphin is a separate GPL-2.0-or-later program. The original MIT bridge sources
and patch-preparation script in `native/` and `scripts/` build a modified Dolphin
locally; the combined executable remains GPL-2.0-or-later. Upstream source,
dependencies, and built binaries are excluded from this Git repository. One
Windows build is distributed separately as a GitHub release asset
(`dolphin-2609a-native-v1`) under GPL-2.0-or-later, together with its complete
corresponding source (the exact patched source tree with all submodules) and the
patch against upstream commit `409881c34357d0f105fd473167d15ab0bd9c1628`. Its
bundled Qt 6.8.3 libraries are used under LGPL-3.0-only; their license texts
are included in the binary archive. See the
[native build/protocol guide](docs/native-backend.md). Installed Python dependencies also retain their own
licenses. Consult each dependency's distribution metadata and upstream license
when redistributing an environment; the project MIT license does not replace
those terms.

The model-free Rainbow/IQN code is original project code implementing published
methods. No DQN Zoo/JAX source was copied into this MIT repository. TensorBoard
is an installed dependency with its own Apache-2.0 license; PyTorch and other
dependencies retain their upstream terms. Local event logs and checkpoints are
excluded from Git, as are the existing emulator binaries and game assets.

The [reference bibliography](docs/references.bib) attributes the research methods
and upstream projects used in the technical documentation. The memory profiles
record address/layout facts derived from pinned Garigari and Petari sources and
local runtime measurements; those projects' game implementations are not
included in this repository. Their licenses and the licenses of referenced
papers and documentation remain with their respective rights holders.
