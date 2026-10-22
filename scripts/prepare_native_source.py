"""Apply the SMG RL bridge to the pinned, separately licensed Dolphin source."""

import argparse
import shutil
import subprocess
from pathlib import Path

REVISION = "409881c34357d0f105fd473167d15ab0bd9c1628"


def prepare(source):
    root = Path(__file__).resolve().parent.parent
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected Dolphin 2609a {REVISION}, found {revision}")
    core_header = (source / "Source/Core/Core/Core.h").read_text(encoding="utf-8")
    core_source = (source / "Source/Core/Core/Core.cpp").read_text(encoding="utf-8")
    if "void DoFrameSteps(" in core_header or ("bool DoFrameSteps(" in core_header and
                                              "SMGRL_PROTOCOL_REVISION_3" not in core_source):
        raise ValueError("Older protocol patches detected. Use a fresh pinned checkout for v3; "
                         "this script does not overwrite existing source changes.")

    def replace(file, old, new):
        path = source / "Source/Core" / file
        content = path.read_text(encoding="utf-8")
        if new in content:
            return
        if old not in content:
            raise ValueError(f"Patch context missing: {file}")
        path.write_text(content.replace(old, new, 1), encoding="utf-8")

    shutil.copyfile(root / "native/SMGRLBridge.h", source / "Source/Core/Common/SMGRLBridge.h")
    shutil.copyfile(root / "native/SMGRLImage.h", source / "Source/Core/Common/SMGRLImage.h")
    shutil.copyfile(root / "native/SMGRLHost.cpp", source / "Source/Core/DolphinQt/SMGRLHost.cpp")
    replace("DolphinQt/CMakeLists.txt", "  MainWindow.cpp", "  SMGRLHost.cpp\n  MainWindow.cpp")
    replace(
        "DolphinQt/MainWindow.cpp",
        "MainWindow::MainWindow(",
        "namespace SMGRL { void Start(QObject*, Core::System&); }\n\nMainWindow::MainWindow(",
    )
    replace(
        "DolphinQt/MainWindow.cpp",
        "  CreateComponents();",
        "  SMGRL::Start(this, m_system);\n  CreateComponents();",
    )
    replace(
        "Core/Core.h",
        "void DoFrameStep(Core::System& system);",
        "void DoFrameStep(Core::System& system);\nbool DoFrameSteps(Core::System& system, unsigned int count);",
    )
    replace(
        "DolphinQt/Settings.cpp",
        '#include "DolphinQt/Settings.h"',
        '#include "DolphinQt/Settings.h"\n#include "Common/SMGRLBridge.h"',
    )
    replace(
        "DolphinQt/Settings.cpp",
        "  m_core_state_changed_hook = Core::AddOnStateChangedCallback([this](Core::State new_state) {",
        """  m_core_state_changed_hook = Core::AddOnStateChangedCallback([this](Core::State new_state) {
#ifdef _WIN32
    // Native batches are already observed through the bridge. Do not enqueue
    // hidden menus/dialog redraws after every batch; retain all Core listeners.
    if (auto* f = SMGRL::Fields(); f && (f[SMGRL::Options] & 8) &&
        new_state == Core::State::Paused && f[SMGRL::Command] == 2 &&
        f[SMGRL::StepComplete] == f[SMGRL::Request] && f[SMGRL::Request] != f[SMGRL::Response])
      return;
#endif""",
    )
    replace(
        "Core/Core.cpp",
        '#include "Core/Core.h"',
        '#include "Core/Core.h"\n#include "Common/SMGRLBridge.h"',
    )
    replace(
        "Core/Core.cpp",
        "static bool s_frame_step = false;",
        "static bool s_frame_step = false;\n// SMGRL_PROTOCOL_REVISION_3\nstatic unsigned int s_native_steps_remaining = 0;\nstatic LONG s_native_request = 0;\nstatic long long s_native_started = 0;",
    )
    replace(
        "Core/Core.cpp",
        "void Callback_NewField(Core::System& system)\n{",
        """void Callback_NewField(Core::System& system)
{
#ifdef _WIN32
  if (auto* f = SMGRL::Fields())
    InterlockedIncrement(&f[SMGRL::FieldCount]);
#endif""",
    )
    replace(
        "Core/Core.cpp",
        "    AsyncRequests::GetInstance()->WaitForEmptyQueue();",
        """    const auto wait_started = SMGRL::Ticks();
    AsyncRequests::GetInstance()->WaitForEmptyQueue();
    if (auto* f = SMGRL::Fields())
    {
      InterlockedIncrement(&f[SMGRL::QueueDrainCalls]);
      InterlockedExchangeAdd(&f[SMGRL::FrameWaitMicros], SMGRL::ElapsedMicros(wait_started));
    }""",
    )
    replace(
        "Core/Core.cpp",
        "      s_frame_step = false;\n      system.GetCPU().Break();",
        """#ifdef _WIN32
      if (s_native_steps_remaining > 0)
      {
        if (auto* f = SMGRL::Fields())
          InterlockedIncrement(&f[SMGRL::CompletedFrames]);
        if (--s_native_steps_remaining > 0)
        {
          s_stop_frame_step.store(false);
          return;
        }
      }
#endif
      s_frame_step = false;
      system.GetCPU().Break();
#ifdef _WIN32
      if (auto* f = SMGRL::Fields(); f && s_native_request)
      {
        f[SMGRL::CpuBatchMicros] = SMGRL::ElapsedMicros(s_native_started);
        InterlockedExchange(&f[SMGRL::StepComplete], s_native_request);
        SMGRL::SignalCpu();
      }
#endif""",
    )
    replace(
        "Core/Core.cpp",
        "void UpdateInputGate(bool require_focus, bool require_full_focus)",
        """bool DoFrameSteps(Core::System& system, unsigned int count)
{
  std::lock_guard lock(s_core_mutex);
  if (GetState(system) != State::Paused || count == 0)
    return false;
  if (auto* f = SMGRL::Fields())
    s_native_request = f[SMGRL::Request];
  s_native_steps_remaining = count;
  s_native_started = SMGRL::Ticks();
  s_stop_frame_step = false;
  s_frame_step = true;
  SetState(system, State::Running, false);
  return true;
}

void UpdateInputGate(bool require_focus, bool require_full_focus)""",
    )
    replace(
        "VideoCommon/FrameDumper.cpp",
        '#include "VideoCommon/FrameDumper.h"',
        '#include "VideoCommon/FrameDumper.h"\n#include "Common/SMGRLBridge.h"',
    )
    replace(
        "VideoCommon/FrameDumper.cpp",
        "  // We only need to render a copy if we need to stretch/scale the XFB copy.",
        """#ifdef _WIN32
  // Optional GPU reduction uses the renderer's linear scaler. The default
  // compact CPU path retains area averaging of the original dump pixels.
  if (auto* f = SMGRL::Fields(); f && f[SMGRL::CaptureMode] == 2)
  {
    const int size = f[SMGRL::ImageSize];
    const int left = src_rect.left + static_cast<int>(
        static_cast<long long>(source_width) * f[SMGRL::CropLeft] / 1000000);
    const int top = src_rect.top + static_cast<int>(
        static_cast<long long>(source_height) * f[SMGRL::CropTop] / 1000000);
    const int right = src_rect.left + static_cast<int>(
        static_cast<long long>(source_width) * f[SMGRL::CropRight] / 1000000);
    const int bottom = src_rect.top + static_cast<int>(
        static_cast<long long>(source_height) * f[SMGRL::CropBottom] / 1000000);
    if (size < 1 || size > 512 || left >= right || top >= bottom)
    { f[SMGRL::Error] = 2; return; }
    if (!CheckFrameDumpRenderTexture(size, size) || !CheckFrameDumpReadbackTexture(size, size))
      return;
    const auto started = SMGRL::Ticks();
    g_gfx->ScaleTexture(m_frame_dump_render_framebuffer.get(),
                       m_frame_dump_render_framebuffer->GetRect(), src_texture,
                       MathUtil::Rectangle<int>(left, top, right, bottom));
    m_frame_dump_readback_texture->CopyFromTexture(m_frame_dump_render_texture.get());
    f[SMGRL::ScaleMicros] = SMGRL::ElapsedMicros(started);
    const auto readback_started = SMGRL::Ticks();
    auto& output = m_frame_dump_readback_texture;
    output->Flush();
    if (output->Map())
    {
      SMGRL::PublishPixels(reinterpret_cast<const unsigned char*>(output->GetMappedPointer()),
                          size, size, static_cast<int>(output->GetMappedStride()));
      output->Unmap();
    }
    f[SMGRL::ReadbackMicros] = SMGRL::ElapsedMicros(readback_started);
    return;
  }
#endif
  // We only need to render a copy if we need to stretch/scale the XFB copy.""",
    )
    replace(
        "VideoCommon/FrameDumper.cpp",
        "  m_last_frame_state = m_ffmpeg_dump.FetchState(ticks, frame_number);",
        """#ifdef _WIN32
  if (SMGRL::Enabled())
  {
    const auto readback_started = SMGRL::Ticks();
    auto& output = m_frame_dump_readback_texture;
    output->Flush();
    if (output->Map())
    {
      SMGRL::PublishPixels(reinterpret_cast<const unsigned char*>(output->GetMappedPointer()),
                          target_width, target_height, static_cast<int>(output->GetMappedStride()));
      output->Unmap();
    }
    SMGRL::Fields()[SMGRL::ReadbackMicros] = SMGRL::ElapsedMicros(readback_started);
    return;
  }
#endif
  m_last_frame_state = m_ffmpeg_dump.FetchState(ticks, frame_number);""",
    )
    replace(
        "VideoCommon/FrameDumper.cpp",
        "bool FrameDumper::IsFrameDumping() const\n{",
        """bool FrameDumper::IsFrameDumping() const
{
#ifdef _WIN32
  if (SMGRL::WantsPixels())
    return true;
#endif""",
    )
    replace(
        "VideoCommon/Present.cpp",
        '#include "VideoCommon/Present.h"',
        '#include "VideoCommon/Present.h"\n#include "Common/SMGRLBridge.h"',
    )
    replace(
        "VideoCommon/Present.cpp",
        "  m_present_count++;",
        "  SMGRL::ScopedMicros present_timer(SMGRL::PresentMicros);\n  m_present_count++;",
    )
    replace(
        "VideoCommon/Present.cpp",
        "  m_present_count++;\n\n  if (g_gfx->IsHeadless()",
        """  m_present_count++;
#ifdef _WIN32
  if (SMGRL::SkipPresentation() && m_xfb_entry)
  {
    g_vertex_manager->Flush();
    UpdateDrawRectangle();
    if (present_info)
    {
      present_info->actual_present_time = Clock::now();
      present_info->present_time_accuracy = PresentInfo::PresentTimeAccuracy::Unimplemented;
    }
    InterlockedIncrement(&SMGRL::Fields()[SMGRL::SkippedPresents]);
    return;
  }
#endif

  if (g_gfx->IsHeadless()""",
    )
    file = "InputCommon/ControllerInterface/DualShockUDPClient/DualShockUDPClient.cpp"
    replace(
        file,
        '#include "InputCommon/ControllerInterface/DualShockUDPClient/DualShockUDPClient.h"',
        '#include "InputCommon/ControllerInterface/DualShockUDPClient/DualShockUDPClient.h"\n#include "Common/SMGRLBridge.h"',
    )
    replace(
        file,
        "Core::DeviceRemoval Device::UpdateInput()\n{",
        """Core::DeviceRemoval Device::UpdateInput()
{
#ifdef _WIN32
  if (auto* f = SMGRL::Fields())
  {
    // Sample the command's input mask inside Dolphin's input update, not over UDP.
    const LONG mask = f[SMGRL::Buttons];
    ResetPadData();
    m_pad_data.button_ps = (mask & 16) ? 1 : 0;
    m_pad_data.button_touch = (mask & 32) ? 1 : 0;
    m_pad_data.button_r1_analog = (mask & 64) ? 255 : 0;
    m_pad_data.left_stick_x = (mask & 8) ? 255 : (mask & 2) ? 0 : 128;
    m_pad_data.left_stick_y_inverted = (mask & 1) ? 0 : (mask & 4) ? 255 : 128;
    m_pad_data.accelerometer_y_g = -1.0f;
    return Core::DeviceRemoval::Keep;
  }
#endif""",
    )
    replace(
        "Common/FileUtil.cpp",
        '#include "Common/FileUtil.h"',
        '#include "Common/FileUtil.h"\n#include "Common/SMGRLBridge.h"',
    )
    replace(
        "Common/FileUtil.cpp",
        "  std::filesystem::rename(StringToPath(srcFilename), StringToPath(destFilename), error);",
        """  std::filesystem::rename(StringToPath(srcFilename), StringToPath(destFilename), error);
#ifdef _WIN32
  // A transient Windows file lock must not strand a hidden worker behind a
  // NAND FST rename dialog. Only explicitly enabled bridge processes retry.
  for (int attempt = 0; error && SMGRL::Enabled() && attempt < 50 &&
       (error.value() == ERROR_ACCESS_DENIED || error.value() == ERROR_SHARING_VIOLATION); ++attempt)
  {
    Sleep(10);
    std::filesystem::rename(StringToPath(srcFilename), StringToPath(destFilename), error);
  }
#endif""",
    )
    print(f"Prepared native bridge at {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    prepare(parser.parse_args().source)
