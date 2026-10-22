// Copyright (c) 2026 PedroM2626
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace SMGRL
{
// Separable area averaging of rounded grayscale pixels, matching the Python
// RGBA2GRAY -> INTER_AREA pipeline to integer-rounding tolerance. GPU linear
// reduction is a separately selected experiment, never an implicit fallback.
class AreaImage
{
  struct Weight { int index; float value; };
  using Axis = std::vector<std::vector<Weight>>;
  int m_width = 0, m_height = 0, m_size = 0;
  int m_left = 0, m_top = 0, m_right = 0, m_bottom = 0;
  Axis m_x, m_y;
  std::vector<unsigned char> m_gray;
  std::vector<float> m_horizontal;

  static Axis MakeAxis(int length, int size)
  {
    Axis result(size);
    const double scale = static_cast<double>(length) / size;
    for (int destination = 0; destination < size; ++destination)
    {
      const double left = destination * scale, right = (destination + 1) * scale;
      for (int source = static_cast<int>(std::floor(left));
           source < std::min(length, static_cast<int>(std::ceil(right))); ++source)
      {
        const double coverage = std::min(right, source + 1.0) - std::max(left, double(source));
        if (coverage > 0)
          result[destination].push_back({source, static_cast<float>(coverage / scale)});
      }
    }
    return result;
  }

public:
  static unsigned char Gray(const unsigned char* p)
  {
    return static_cast<unsigned char>((9798 * p[0] + 19235 * p[1] + 3735 * p[2] + 16384) >> 15);
  }

  void Convert(const unsigned char* pixels, int width, int height, int stride,
               int left, int top, int right, int bottom, int size, unsigned char* output)
  {
    if (m_width != width || m_height != height || m_size != size ||
        m_left != left || m_top != top || m_right != right || m_bottom != bottom)
    {
      m_width = width; m_height = height; m_size = size;
      m_left = left; m_top = top; m_right = right; m_bottom = bottom;
      m_x = MakeAxis(right - left, size);
      m_y = MakeAxis(bottom - top, size);
      m_gray.resize((right - left) * (bottom - top));
      m_horizontal.resize((bottom - top) * size);
    }
    const int crop_width = right - left, crop_height = bottom - top;
    for (int y = 0; y < crop_height; ++y)
      for (int x = 0; x < crop_width; ++x)
        m_gray[y * crop_width + x] = Gray(pixels + (top + y) * stride + (left + x) * 4);
    for (int y = 0; y < crop_height; ++y)
      for (int x = 0; x < size; ++x)
      {
        float value = 0;
        for (const auto& weight : m_x[x])
          value += m_gray[y * crop_width + weight.index] * weight.value;
        m_horizontal[y * size + x] = value;
      }
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < size; ++x)
      {
        float value = 0;
        for (const auto& weight : m_y[y])
          value += m_horizontal[weight.index * size + x] * weight.value;
        output[y * size + x] = static_cast<unsigned char>(std::clamp(std::lrint(value), 0L, 255L));
      }
  }
};
}  // namespace SMGRL
