# Verified SB4E01 memory profile

All addresses below are **emulated Wii addresses**, not host process pointers. Wii data is big-endian. RAM is read through `dolphin-memory-engine`; the project never writes game RAM or invokes emulated functions.

The profile was derived from the live executable and the [Garigari SB4E01 symbol map](https://github.com/SMGCommunity/Garigari/blob/89df1f33c7f6bcf1c159c9f91b0ad76b04a1d84e/config/SB4E01/symbols.txt). Only address/layout facts are used; game code and assets are not distributed.

## Pointer resolution

Here `U32(address)` means a big-endian unsigned 32-bit read:

```text
game_system      = U32(0x807D0DA4)
scene_controller = U32(game_system + 0x24)
scene            = U32(scene_controller + 0x12C)
scene_objects    = U32(scene + 0x10)
mario_holder     = U32(scene_objects + 0x54)  # scene object index 0x15
mario_actor      = U32(mario_holder + 0x14)
actor_spine      = U32(mario_actor + 0x58)
scene_spine      = U32(scene + 0x04)
```

The chain is resolved on every sample. Loading another state can relocate the actor. A single hardcoded actor address is not safe, even between the two variants.

## Values

| Signal | Location / interpretation | Evidence |
| --- | --- | --- |
| Game ID | Six ASCII bytes at `0x80000000` must be `SB4E01` | Live |
| Executable layout | Bytes at `0x8045A970` must begin `80 6D 9A 84 80 63 00 24` | Live code / `getSceneObjHolder` chain |
| Scene identity | `U32(scene)` must be GameScene vtable `0x806EFA28` | Symbol and live object |
| Stage | ASCII string at `scene_controller + 0x40` must be `SecretAthleticGalaxy` | Both variations live |
| Scenario | `U32(scene_controller + 0x80)` is 1 or 2 | All four states live |
| Mario X / Y / Z | Three contiguous big-endian floats at `mario_actor + 0x14`, `+0x18`, `+0x1C` | Movement and star/death tests |
| Actor state | `current_nerve(actor_spine)` | Live transitions |
| Scene state | `current_nerve(scene_spine)` | Live transitions |
| Scene frame counter | `U32(scene_spine + 0x0C)` | Changes while running, stable while paused |
| Completion | Scene state equals `0x807D5318` | Both stars collected live |
| Death by abyss | Actor state equals `0x807D4E3C` | Both start states tested live |
| Game over | Scene state equals `0x807D5320` | Both fall tests live |
| Other forced death | Actor state equals `0x807D4E38` | Derived from `MarioActor::forceKill`; not exercised by the fall test |
| Grand Star completion | Scene state equals `0x807D531C` | Symbol/code-derived; not used by these scenarios |

`current_nerve(spine)` returns `U32(spine + 8)` if nonzero; otherwise `U32(spine + 4)`. The pending state has priority, matching `Spine::getCurrentNerve` at `0x802432A0`.

The completion singleton is a **state identity**, not the value stored at an address named "completion". Compare the object's current nerve pointer to `0x807D5318`; reading the byte at `0x807D5318` as a Boolean would be incorrect. Normal gameplay uses scene nerve `0x807D5310`.

## Progress and coordinate frames

Both starts are near `(0, -1150, 500)`. The first near-star point is near `(-2835, -450, -37225)`. The second paused snapshot is near `(-3095, -695, -52697)`. Earlier captures taken after settling on the rotating platform were near `(-2857, -627, -52697)`; the final profile restores states while paused to preserve the saved position. Use the current `config/calibration.json` rather than assuming an immutable heap address or platform-relative position.

For each variation, project the current X/Z position onto its own start-to-goal vector, clamp to `[0, 1]`, and reward only increases over the episode's farthest valid projection. Reversing and re-crossing the same point cannot repeatedly earn progress. A large one-step displacement, a height below `min_reward_y`, or death suppresses progress reward. The policy never receives this projection.

There is no claim of a discovered native course percentage field. This progress value is computed from the verified coordinate addresses. It is appropriate to these mostly longitudinal courses; a looping course would need ordered waypoints or a route graph.

The top-down plot uses world X horizontally and world Z vertically with equal coordinate scale. World Y is height, retained in CSV for fall inspection. The projection is a world-space trace, not a collision map and not a camera-relative minimap.

## Revalidation

Run `smg-rl calibrate` with no other Dolphin process, or `smg-rl calibrate --attach` while a compatible session is already open. The command rejects a near-star baseline already marked complete, then requires a real post-jump completion and a real post-fall death in **each** variation. Inspect `config/calibration.json` and the emitted CSV/PNG evidence. Never label an untested regional address map as verified.
