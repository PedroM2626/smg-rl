// Copyright (c) 2026 PedroM2626
// SPDX-License-Identifier: MIT
// Loaded only when an explicitly named, versioned shared-memory mapping exists.
#include "Common/SMGRLBridge.h"

#ifdef _WIN32
#include <QCoreApplication>
#include <QObject>
#include <QTimer>

#include "Core/Core.h"
#include "Core/State.h"
#include "Core/System.h"

namespace SMGRL
{
void Start(QObject* parent, Core::System& system)
{
  auto* f = Fields();
  if (!f)
    return;
  auto* timer = new QTimer(parent);
  timer->setTimerType(Qt::PreciseTimer);
  // Host-thread state access and savestate loading follow Dolphin's own UI path.
  QObject::connect(timer, &QTimer::timeout, parent,
                   [f, &system, pending = LONG{0}, pending_command = LONG{0},
                    pending_argument = LONG{0}]() mutable {
    const auto state = Core::GetState(system);
    f[Paused] = state == Core::State::Paused;
    if (pending)
    {
      if (state != Core::State::Paused)
        return;
      if (f[Command] == 3 && f[LoadComplete] != pending)
        return;
      if (pending_command == 2 &&
          (f[StepComplete] != pending || f[CompletedFrames] != pending_argument))
        return;
      // GetState(Paused) is not a CPU-idle barrier. SetState waits for the CPU
      // and adjacent systems to stop before publishing a completed operation.
      Core::SetState(system, Core::State::Paused, false);
      f[Paused] = 1;
      f[AcknowledgedField] = f[FieldCount];
      const LONG completed = pending;
      pending = 0;
      MemoryBarrier();
      InterlockedExchange(&f[Response], completed);
      return;
    }
    const LONG request = f[Request];
    if (request == f[Response] || !Core::IsRunning(system))
      return;
    MemoryBarrier();
    f[Error] = 0;
    const LONG command = f[Command];
    pending_command = command;
    pending_argument = f[Argument];
    if (command == 1 || command == 4)
    {
      Core::SetState(system, Core::State::Paused);
      pending = request;
    }
    else if (command == 2)
    {
      if (pending_argument < 1 || pending_argument > 600)
      {
        f[Error] = 1;
        InterlockedExchange(&f[Response], request);
        return;
      }
      // A transient running flag is reconciled only if no field has advanced
      // since the previous acknowledgment. Never hide unobserved gameplay.
      if (state != Core::State::Paused)
        InterlockedIncrement(&f[Reconciliations]);
      Core::SetState(system, Core::State::Paused, false);
      if (f[FieldCount] != f[AcknowledgedField])
      {
        f[Error] = 4;
        f[RejectedState] = static_cast<LONG>(state);
        f[Paused] = 1;
        InterlockedExchange(&f[Response], request);
        return;
      }
      f[CompletedFrames] = 0;
      f[FrameWaitMicros] = 0;
      pending = request;
      if (!Core::DoFrameSteps(system, static_cast<unsigned int>(pending_argument)))
      {
        pending = 0;
        f[Error] = 3;
        f[RejectedState] = static_cast<LONG>(Core::GetState(system));
        InterlockedExchange(&f[Response], request);
      }
    }
    else if (command == 3)
    {
      if (f[Argument] < 1 || f[Argument] > 4)
      {
        f[Error] = 1;
        InterlockedExchange(&f[Response], request);
        return;
      }
      Core::SetState(system, Core::State::Paused);
      State::Load(system, static_cast<unsigned int>(f[Argument]));
      // State::Load queues asynchronous CPU work. A paused-state observation
      // alone is not an acknowledgment that the savestate was restored.
      Core::RunOnCPUThread(system, [f, request] {
        MemoryBarrier();
        InterlockedExchange(&f[LoadComplete], request);
      });
      Core::SetState(system, Core::State::Paused);
      pending = request;
    }
    else
    {
      f[Error] = 1;
      InterlockedExchange(&f[Response], request);
    }
  });
  timer->start(1);
}
}  // namespace SMGRL
#endif
