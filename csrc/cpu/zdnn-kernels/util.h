// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <torch/all.h>
#include <zdnn.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fenv.h>

namespace {

inline void zdnn_check(zdnn_status status, const char* what) {
  // Range violation is a warning: the operation still wrote the output.
  if (status == ZDNN_OK ||
      (static_cast<uint32_t>(status) & ZDNN_WARNING) != 0) {
    return;
  }
  const char* message = zdnn_get_status_message(status);
  TORCH_CHECK(false, "zDNN ", what,
              " failed: ", message != nullptr ? message : "unknown status");
}

// Software unstick writes every element, then returns ZDNN_CONVERT_FAILURE
// when a DLFLOAT16 value does not fit the host dtype. The buffer holds inf
// for those elements. Warmup activations hit this; aborting kills engine init.
inline void unstick(const zdnn_ztensor* z, void* dest) {
  const zdnn_status status = zdnn_transform_origtensor(z, dest);
  if (status == ZDNN_OK ||
      (static_cast<uint32_t>(status) & ZDNN_WARNING) != 0) {
    return;
  }
  if (status == ZDNN_CONVERT_FAILURE) {
    feclearexcept(FE_ALL_EXCEPT);
    TORCH_WARN_ONCE(
        "zDNN linear output exceeded the host dtype; overflowed elements "
        "were stored as inf.");
    return;
  }
  zdnn_check(status, "transform_origtensor");
}

inline zdnn_data_types host_type(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return FP32;
    case at::kHalf:
      return FP16;
    case at::kBFloat16:
      return BFLOAT;
    default:
      TORCH_CHECK(false, "zDNN linear supports fp32, fp16, and bf16");
  }
}

inline size_t host_elem_size(zdnn_data_types type) {
  switch (type) {
    case FP32:
      return 4;
    case FP16:
    case BFLOAT:
      return 2;
    default:
      return 0;
  }
}

inline uint32_t dim_limit(uint8_t dimension, uint32_t mdis) {
  const uint32_t specific = zdnn_get_max_for_dim(dimension);
  return specific == 0 ? mdis : specific;
}

// Outermost dimension first, matching zdnn_init_pre_transformed_desc.
inline bool transformed_fits(zdnn_data_layouts layout, zdnn_data_types type,
                             uint32_t dim_a, uint32_t dim_b, uint64_t mts) {
  zdnn_tensor_desc pre{};
  zdnn_tensor_desc tfm{};
  if (layout == ZDNN_1D) {
    zdnn_init_pre_transformed_desc(layout, type, &pre, dim_a);
  } else {
    zdnn_init_pre_transformed_desc(layout, type, &pre, dim_a, dim_b);
  }
  if (zdnn_generate_transformed_desc(&pre, &tfm) != ZDNN_OK) {
    return false;
  }
  return zdnn_getsize_ztensor(&tfm) <= mts;
}

struct ZdnnTensor {
  zdnn_tensor_desc pre{};
  zdnn_tensor_desc tfm{};
  zdnn_ztensor z{};
  bool allocated = false;

  ZdnnTensor() = default;
  ZdnnTensor(const ZdnnTensor&) = delete;
  ZdnnTensor& operator=(const ZdnnTensor&) = delete;

  ~ZdnnTensor() { release(); }

  void release() {
    if (allocated) {
      zdnn_free_ztensor_buffer(&z);
      allocated = false;
      z = {};
      pre = {};
      tfm = {};
    }
  }

  void load(zdnn_data_layouts layout, zdnn_data_types type, uint32_t dim_a,
            uint32_t dim_b, const void* data, size_t elem_size,
            bool saturate = false) {
    release();
    if (layout == ZDNN_1D) {
      zdnn_init_pre_transformed_desc(layout, type, &pre, dim_a);
    } else {
      zdnn_init_pre_transformed_desc(layout, type, &pre, dim_a, dim_b);
    }
    zdnn_check(zdnn_generate_transformed_desc(&pre, &tfm),
               "generate_transformed_desc");
    zdnn_check(zdnn_init_ztensor_with_malloc(&pre, &tfm, &z),
               "init_ztensor_with_malloc");
    allocated = true;
    transform_from(data, elem_size, dim_a, layout == ZDNN_1D ? 1 : dim_b,
                   saturate);
  }

  // Host data changed; the stick buffer is reused.
  void reload(const void* data, size_t elem_size, uint32_t dim_a,
              uint32_t dim_b, bool saturate = false) {
    zdnn_reset_ztensor(&z);
    transform_from(data, elem_size, dim_a, dim_b, saturate);
  }

 private:
  void transform_from(const void* data, size_t elem_size, uint32_t dim_a,
                      uint32_t dim_b, bool saturate) {
    if (data != nullptr) {
      // Activations can exceed the DLFLOAT16 range during warmup. Saturation
      // clamps those values instead of aborting the forward.
      const zdnn_status status =
          saturate ? zdnn_transform_ztensor_with_saturation(&z, data)
                   : zdnn_transform_ztensor(&z, data);
      zdnn_check(status, "transform_ztensor");
      return;
    }
    const size_t bytes =
        static_cast<size_t>(dim_a) * static_cast<size_t>(dim_b) * elem_size;
    // Hardware stickify reads this buffer as a tensor, so it is 4K aligned.
    size_t cap = (bytes + 4095u) & ~size_t{4095};
    if (cap == 0) {
      cap = 4096;
    }
    void* zeros = nullptr;
    TORCH_CHECK(posix_memalign(&zeros, 4096, cap) == 0,
                "zDNN aligned allocation failed");
    std::memset(zeros, 0, bytes);
    const zdnn_status status = zdnn_transform_ztensor(&z, zeros);
    std::free(zeros);
    zdnn_check(status, "transform_ztensor");
  }
};

}  // namespace
