# Kernel pool growth: audio-backend correction

## Reproduced mechanism

On October 29, elevated Windows diagnostics identified a concrete configuration
fault in the shared background/native worker setup: `[DSP] Backend = Null` is
not Dolphin's null-output identifier. The retained Dolphin source defines
`BACKEND_NULLSOUND` as `No Audio Output` in
`Source/Core/Core/Config/MainSettings.h`. `AudioCommon.cpp` falls back to the
default backend for an unknown identifier. On this build the default is Cubeb.
Setting volume to zero therefore muted a real audio stream rather than removing it.

`CPUManager::RunAdjacentSystems` starts/stops the sound stream on pause/resume.
The native stepping path repeatedly makes those transitions, including when
state-change notifications are suppressed. This explains why disabling UI
notifications or simply muting audio did not remove the trigger.

A bounded Windows Performance Recorder Pool trace captured 30 pause/resume
cycles in the installed stock Dolphin, using an isolated copied profile and
the canonical SMG1 state, without a learner or policy actions. Allocation stacks
included `MMDevAPI.dll` called by both `audiosrv.dll` and
`RtkAudUService64.exe`, continuing into `ntdll.dll` and the kernel. The selected
trace extraction contains module offsets, not resolved function names. It
attributes **new allocations during that test**, not every byte of the older
multi-GiB accumulation or a specific defective vendor function.

## Correction and matched check

`prepare_worker` now writes `Backend = No Audio Output`. This applies to both
games and both background/native workers the next time they are prepared.
Existing source emulator settings and retained experiment profiles are untouched.
Reward settings, action timing, checkpoints and replay were not changed. Future
collection with the correction must record that the emulator audio configuration
changed; it is not a bitwise reproduction of earlier sampling.

| Non-learning probe | Pause/resume cycles | EtwD growth | Etwr growth | Combined |
| --- | ---: | ---: | ---: | ---: |
| Invalid `Null`, no WPR | 30 | 296,592 B | 203,360 B | 499,952 B |
| Invalid `Null`, WPR trace | 30 | 270,672 B | 185,440 B | 456,112 B |
| Correct `No Audio Output`, no WPR | 30 | 176 B | 160 B | 336 B |

The corrected short check reduced observed combined growth by 99.93% relative
to the untraced invalid-backend check. It is a system-wide counter comparison,
not an assertion of zero background allocations or a completed multi-hour
six-worker validation. The WPR trace itself used substantial temporary buffers;
its recording and merge finished successfully and those buffers were released.

The modified native Dolphin executable was subsequently rejected by Windows
Application Control (`WinError 4551`, Code Integrity event 3077). The original
installed Dolphin remained executable and was used for the independent
pause/resume diagnosis. No signing policy, security setting or trust barrier
was changed. The corrected native six-worker path has therefore **not** been
revalidated in this session, and no training was resumed.

## Other hypotheses and interventions

- A three-phase, 90-second native probe temporarily suspended and restored the
  verified Logitech user agent. Pool growth persisted during suspension; that
  intervention did not identify the cause. The agent was restored in `finally`.
- Two Acer services held over 1.2 million process handles in aggregate. A scoped
  restart reduced the counts to hundreds while preserving their automatic start
  modes and running state. Counts started rising again, and the ETW accumulation
  was not released. This is a separate unresolved handle-growth finding.
- Stopping only the optional `ULIC Monitor FPS` ETW session did not reclaim the
  accumulation. The later invalid-audio pause test still grew with that session
  absent. Its name/vendor resemblance to PredatorSense does not establish that
  PredatorSense caused the leak. No thermal/fan configuration was changed.
- Fifty short-lived Python processes, twenty Vulkan instance lifecycles, 2,000
  submitted Vulkan command buffers, 100,000 event signals and 100,000 self-memory
  reads did not reproduce growth in the two tags in their short isolated checks.
  These checks do not exonerate every driver or cross-process access pattern.
- Stock Dolphin running freely for 45 seconds added 16,336 B combined, far less
  than repeated pause/resume. It was a boot diagnostic, not a matched gameplay
  benchmark.

## Evidence, scope and next validation

Raw evidence remains local under
`.local/kernel-memory-investigation-20261029/`, including the WPR ETL, selected
allocation/stack extraction, pool snapshots, service interventions and failed
native startup. These local files contain system diagnostics and are not committed.
The committed compact audit retains the measurements and selected module stacks.
No ROM, state, profile, replay or checkpoint is included in Git.

The previously accumulated kernel memory is distinct from stopping new growth.
Do not label a service restart or the short corrected check as proof that the
old allocation has been reclaimed. A final controlled restart of Windows Audio,
the Realtek service and its surviving user/system helpers preserved automatic
startup modes and restored both services to Running; the user helper was
relaunched without elevation. It **did not reclaim** approximately 12.1 GiB
still charged to EtwD/Etwr. The first cleanup helper failed on a PowerShell
hashtable-property expansion; a second waited on an internal confirmation in
its hidden console. Both faults were retained, the waiting helper was stopped,
and the scoped cleanup/restoration completed using a non-interactive helper.
No diagnostic recording, emulator or elevated helper was left running.

Rebooting Windows is the next practical step to clear the retained kernel
allocation; it was not performed automatically because it would interrupt the
owner's open work. This is separate from the native executable's Application
Control rejection and does not promise that rebooting resolves that rejection.
Before another long run, verify a permitted
native build, then perform a bounded stepping health check with pool counters,
RAM/commit guards, actual worker identities and an explicit stop boundary.
Preserve the audited training checkpoint; no new fitting or evaluation was
allocated by these diagnostics.

The [compact final audit](assets/performance/kernel-memory-audio-fix-20261029.json)
retains exact counters, selected module stacks and hashes of the local raw
evidence. Portable validation passed 22 background/native/pool tests and Ruff.

Microsoft documents pool tags as allocation accounting, and recommends pool
allocation stacks for attribution:
[PoolMon leak diagnosis](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/using-poolmon-to-find-a-kernel-mode-memory-leak),
[WPR/WPA troubleshooting](https://learn.microsoft.com/en-us/troubleshoot/windows-server/support-tools/support-tools-xperf-wpa-wpr).
The Dolphin backend identifier and fallback were checked against the retained
source, rather than inferred from a pool tag or a forum suggestion.
