// Copyright (c) 2026 PedroM2626
// SPDX-License-Identifier: MIT
#pragma once

#ifdef _WIN32
#include <Windows.h>
#include <cstdlib>
#include <cstring>

namespace SMGRL
{
constexpr int HEADER_BYTES = 256;
constexpr int PIXEL_BYTES = 1920 * 1080 * 4;
enum Field : int
{
  Magic, Version, Pid, Request, Response, Command, Argument, Buttons,
  CompletedFrames, ImageSequence, Width, Height, Stride, Error, Paused, LoadComplete
};

struct Mapping
{
  HANDLE handle = nullptr;
  volatile LONG* fields = nullptr;
  Mapping()
  {
    const char* name = std::getenv("SMG_RL_BRIDGE");
    if (!name || !*name)
      return;
    handle = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name);
    if (handle)
      fields = static_cast<volatile LONG*>(
          MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, HEADER_BYTES + PIXEL_BYTES));
    if (fields && fields[Magic] == 0x534d4752 && fields[Version] == 1)
      fields[Pid] = static_cast<LONG>(GetCurrentProcessId());
    else
      fields = nullptr;
  }
  ~Mapping()
  {
    if (fields)
      UnmapViewOfFile(const_cast<LONG*>(fields));
    if (handle)
      CloseHandle(handle);
  }
};

inline volatile LONG* Fields()
{
  static Mapping mapping;
  return mapping.fields;
}

inline bool Enabled() { return Fields() != nullptr; }

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
  for (int y = 0; y < height; ++y)
    std::memcpy(output + y * width * 4, pixels + y * stride, width * 4);
  f[Width] = width;
  f[Height] = height;
  f[Stride] = width * 4;
  MemoryBarrier();
  InterlockedIncrement(&f[ImageSequence]);
}
}  // namespace SMGRL
#endif
