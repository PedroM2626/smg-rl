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
  QObject::connect(timer, &QTimer::timeout, parent, [f, &system, pending = LONG{0}]() mutable {
    const auto state = Core::GetState(system);
    f[Paused] = state == Core::State::Paused;
    if (pending)
    {
      if (state != Core::State::Paused)
        return;
      if (f[Command] == 3 && f[LoadComplete] != pending)
        return;
      MemoryBarrier();
      InterlockedExchange(&f[Response], pending);
      pending = 0;
      return;
    }
    const LONG request = f[Request];
    if (request == f[Response] || !Core::IsRunning(system))
      return;
    MemoryBarrier();
    f[Error] = 0;
    const LONG command = f[Command];
    if (command == 1 || command == 4)
    {
      Core::SetState(system, Core::State::Paused);
      pending = request;
    }
    else if (command == 2)
    {
      if (state != Core::State::Paused || f[Argument] < 1 || f[Argument] > 600)
      {
        f[Error] = 1;
        InterlockedExchange(&f[Response], request);
        return;
      }
      f[CompletedFrames] = 0;
      pending = request;
      Core::DoFrameSteps(system, static_cast<unsigned int>(f[Argument]));
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
