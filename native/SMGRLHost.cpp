// Copyright (c) 2026 PedroM2626
// SPDX-License-Identifier: MIT
// Host state changes follow Dolphin's UI path; events replace latency polling.
#include "Common/SMGRLBridge.h"

#ifdef _WIN32
#include <functional>
#include <memory>
#include <QObject>
#include <QTimer>
#include <QWinEventNotifier>

#include "Core/Core.h"
#include "Core/State.h"
#include "Core/System.h"

namespace SMGRL
{
void Start(QObject* parent, Core::System& system)
{
  auto* f = Fields();
  if (!f) return;
  auto process = std::make_shared<std::function<void()>>();
  *process = [f, &system, pending = LONG{0}, command = LONG{0}, argument = LONG{0}]() mutable {
    f[Paused] = Core::GetState(system) == Core::State::Paused;
    auto complete = [&] {
      if (!pending || Core::GetState(system) != Core::State::Paused) return;
      if (command == 3 && f[LoadComplete] != pending) return;
      if (command == 2 && (f[StepComplete] != pending || f[CompletedFrames] != argument)) return;
      const auto started = Ticks();
      Core::SetState(system, Core::State::Paused, false);
      f[CpuIdleMicros] = ElapsedMicros(started);
      f[Paused] = 1;
      f[AcknowledgedField] = f[FieldCount];
      const LONG sequence = pending;
      pending = 0;
      Reply(sequence);
    };
    if (pending) { complete(); return; }
    const LONG request = f[Request];
    if (request == f[Response] || !Core::IsRunning(system)) return;
    MemoryBarrier();
    const auto requested = static_cast<long long>(
        (static_cast<unsigned long long>(static_cast<unsigned int>(f[RequestTicksHigh])) << 32) |
        static_cast<unsigned int>(f[RequestTicksLow]));
    f[DispatchMicros] = requested ? ElapsedMicros(requested) : 0;
    f[Error] = 0;
    command = f[Command]; argument = f[Argument];
    if (command == 1 || command == 4)
    {
      Core::SetState(system, Core::State::Paused);
      pending = request;
    }
    else if (command == 2)
    {
      if (argument < 1 || argument > 600) { f[Error] = 1; Reply(request); return; }
      const auto state = Core::GetState(system);
      if (state != Core::State::Paused) InterlockedIncrement(&f[Reconciliations]);
      Core::SetState(system, Core::State::Paused, false);
      if (f[FieldCount] != f[AcknowledgedField])
      {
        f[Error] = 4; f[RejectedState] = static_cast<LONG>(state); f[Paused] = 1;
        Reply(request); return;
      }
      f[CompletedFrames] = f[FrameWaitMicros] = f[PresentMicros] = f[ScaleMicros] = 0;
      f[QueueDrainCalls] = f[CpuBatchMicros] = 0;
      pending = request;
      if (!Core::DoFrameSteps(system, static_cast<unsigned int>(argument)))
      {
        pending = 0; f[Error] = 3;
        f[RejectedState] = static_cast<LONG>(Core::GetState(system));
        Reply(request); return;
      }
    }
    else if (command == 3)
    {
      if (argument < 1 || argument > 4) { f[Error] = 1; Reply(request); return; }
      Core::SetState(system, Core::State::Paused);
      State::Load(system, static_cast<unsigned int>(argument));
      Core::RunOnCPUThread(system, [f, request] {
        MemoryBarrier();
        InterlockedExchange(&f[LoadComplete], request);
        SignalCpu();
      });
      Core::SetState(system, Core::State::Paused);
      pending = request;
    }
    else { f[Error] = 1; Reply(request); return; }
    complete();
  };
  // The slow timer handles boot readiness and provides a missed-wakeup safety net.
  const bool events = (f[Options] & 4) && GetMapping().request_event && GetMapping().cpu_event;
  if (events)
    for (HANDLE event : {GetMapping().request_event, GetMapping().cpu_event})
    {
      auto* notifier = new QWinEventNotifier(event, parent);
      QObject::connect(notifier, &QWinEventNotifier::activated, parent,
                       [process](HANDLE) { (*process)(); });
    }
  auto* timer = new QTimer(parent);
  timer->setTimerType(Qt::PreciseTimer);
  QObject::connect(timer, &QTimer::timeout, parent, [process] { (*process)(); });
  timer->start(events ? 50 : 1);
}
}  // namespace SMGRL
#endif
