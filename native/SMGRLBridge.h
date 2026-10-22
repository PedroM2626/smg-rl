// Copyright (c) 2026 PedroM2626
// SPDX-License-Identifier: MIT
#pragma once

#ifdef _WIN32
#include <Windows.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include "SMGRLImage.h"

namespace SMGRL
{
constexpr int HEADER_BYTES = 256;
constexpr int PIXEL_BYTES = 1920 * 1080 * 4;
enum Field : int
{
  Magic, Version, Pid, Request, Response, Command, Argument, Buttons,
  CompletedFrames, ImageSequence, Width, Height, Stride, Error, Paused, LoadComplete,
  StepComplete, FieldCount, AcknowledgedField, Options, ReadbackMicros, FrameWaitMicros,
  SkippedPresents, Reconciliations, RejectedState,
  ImageSize, CropLeft, CropTop, CropRight, CropBottom, PixelFormat,
  CpuBatchMicros, CpuIdleMicros, RequestTicksLow, RequestTicksHigh, DispatchMicros,
  PresentMicros, ScaleMicros, ImageBytes, SourceWidth, SourceHeight,
  QueueDrainCalls, PreprocessMicros, CaptureMode
};

struct Mapping
{
  HANDLE handle = nullptr;
  volatile LONG* fields = nullptr;
  HANDLE request_event = nullptr, reply_event = nullptr, cpu_event = nullptr;
  Mapping()
  {
    const char* name = std::getenv("SMG_RL_BRIDGE");
    if (!name || !*name)
      return;
    handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
    if (handle)
      fields = static_cast<volatile LONG*>(
          MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, HEADER_BYTES + PIXEL_BYTES));
    if (fields && fields[Magic] == 0x534d4752 && fields[Version] == 3)
    {
      fields[Pid] = static_cast<LONG>(GetCurrentProcessId());
      const std::string prefix(name);
      request_event = OpenEventA(SYNCHRONIZE, FALSE, (prefix + "-request").c_str());
      reply_event = OpenEventA(EVENT_MODIFY_STATE, FALSE, (prefix + "-reply").c_str());
      cpu_event = OpenEventA(SYNCHRONIZE | EVENT_MODIFY_STATE, FALSE, (prefix + "-cpu").c_str());
    }
    else
      fields = nullptr;
  }
  ~Mapping()
  {
    for (HANDLE event : {request_event, reply_event, cpu_event})
      if (event) CloseHandle(event);
    if (fields)
      UnmapViewOfFile(const_cast<LONG*>(fields));
    if (handle)
      CloseHandle(handle);
  }
};

inline Mapping& GetMapping()
{
  static Mapping mapping;
  return mapping;
}

inline volatile LONG* Fields() { return GetMapping().fields; }
inline void SignalCpu() { if (GetMapping().cpu_event) SetEvent(GetMapping().cpu_event); }
inline void Reply(LONG sequence)
{
  MemoryBarrier();
  InterlockedExchange(&Fields()[Response], sequence);
  if (GetMapping().reply_event) SetEvent(GetMapping().reply_event);
}

inline bool Enabled() { return Fields() != nullptr; }

inline bool SkipPresentation() { auto* f = Fields(); return f && (f[Options] & 1); }

inline long long Ticks()
{
  LARGE_INTEGER value;
  QueryPerformanceCounter(&value);
  return value.QuadPart;
}

inline LONG ElapsedMicros(long long started)
{
  static const long long frequency = [] { LARGE_INTEGER f; QueryPerformanceFrequency(&f); return f.QuadPart; }();
  return static_cast<LONG>((Ticks() - started) * 1000000 / frequency);
}

class ScopedMicros
{
  Field m_field;
  long long m_started = Ticks();
public:
  explicit ScopedMicros(Field field) : m_field(field) {}
  ~ScopedMicros()
  {
    if (auto* f = Fields(); f && f[Command] == 2 && f[Request] != f[Response])
      InterlockedExchangeAdd(&f[m_field], ElapsedMicros(m_started));
  }
};

inline bool WantsPixels()
{
  auto* f = Fields();
  if (!f)
    return false;
  // Intermediate action frames need rendering, but no CPU readback. The CPU
  // frame-step loop drains the GPU queue at each boundary before incrementing.
  return f[Command] != 2 || f[Request] == f[Response] ||
         f[CompletedFrames] >= f[Argument] - 1;
}

inline void PublishPixels(const unsigned char* pixels, int width, int height, int stride)
{
  auto* f = Fields();
  if (!f)
    return;
  if (width <= 0 || height <= 0 || width > 1920 || height > 1080 || stride < width * 4)
  {
    f[Error] = 2;
    return;
  }
  auto* output = reinterpret_cast<unsigned char*>(const_cast<LONG*>(f)) + HEADER_BYTES;
  f[SourceWidth] = width;
  f[SourceHeight] = height;
  const auto started = Ticks();
  const int size = f[ImageSize];
  if (f[CaptureMode] != 0 && size > 0 && size <= 512)
  {
    if (f[CaptureMode] == 2)
    {
      for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
          output[y * width + x] = AreaImage::Gray(pixels + y * stride + x * 4);
    }
    else
    {
      static thread_local AreaImage resizer;
      const auto coordinate = [](int length, LONG fraction) { return static_cast<int>(
          static_cast<long long>(length) * fraction / 1000000); };
      const int left = coordinate(width, f[CropLeft]), top = coordinate(height, f[CropTop]);
      const int right = coordinate(width, f[CropRight]), bottom = coordinate(height, f[CropBottom]);
      if (left < 0 || top < 0 || right <= left || bottom <= top || right > width || bottom > height)
      { f[Error] = 2; return; }
      resizer.Convert(pixels, width, height, stride, left, top, right, bottom, size, output);
      width = height = size;
    }
    f[PixelFormat] = 1;
    f[Stride] = width;
    f[ImageBytes] = width * height;
  }
  else
  {
    for (int y = 0; y < height; ++y)
      std::memcpy(output + y * width * 4, pixels + y * stride, width * 4);
    f[PixelFormat] = 0;
    f[Stride] = width * 4;
    f[ImageBytes] = width * height * 4;
  }
  f[PreprocessMicros] = ElapsedMicros(started);
  f[Width] = width;
  f[Height] = height;
  MemoryBarrier();
  InterlockedIncrement(&f[ImageSequence]);
}
}  // namespace SMGRL
#endif
