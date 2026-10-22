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

ROMs and savestates are supplied locally by the user and excluded from Git and
the Python distribution. This repository grants no rights to those assets.

## Emulator, research sources, and dependencies

Dolphin is a separate GPL-2.0-or-later program. The original MIT bridge sources
and patch-preparation script in `native/` and `scripts/` build a modified Dolphin
locally; the combined executable remains GPL-2.0-or-later. Upstream source,
dependencies, and built binaries are excluded from this Git repository. See the
[native build/protocol guide](docs/native-backend.md). Installed Python dependencies also retain their own
licenses. Consult each dependency's distribution metadata and upstream license
when redistributing an environment; the project MIT license does not replace
those terms.

The [reference bibliography](docs/references.bib) attributes the research methods
and upstream projects used in the technical documentation. The memory profiles
record address/layout facts derived from pinned Garigari and Petari sources and
local runtime measurements; those projects' game implementations are not
included in this repository. Their licenses and the licenses of referenced
papers and documentation remain with their respective rights holders.
