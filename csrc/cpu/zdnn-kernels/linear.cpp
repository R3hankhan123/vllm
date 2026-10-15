// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// s390x dense linear on NNPA-MATMUL-OP (function code 113).
// libzdnn converts host FP32/FP16/BF16 into NNP-data-type-1 with the
// vector instructions from Chapter 26 (VCRNF for FP32, after BF16 is
// widened to FP32) and stores data-layout-format 0. This file only
// issues zdnn_matmul_op. Attention stays on the VXE kernels.
//
// K or N above the queried dimension or tensor limit is split into tiles.
// Partial products along K are accumulated on the host.

#include "util.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

namespace {

// ZdnnTensor stores pointers to its own descriptors, so it must not move.
struct MatmulTile {
  int64_t k0 = 0;
  int64_t k_len = 0;
  int64_t n0 = 0;
  int64_t n_len = 0;
  ZdnnTensor weight;
  ZdnnTensor bias;
};

// One stick buffer. Activations and outputs are cached separately so an
// N-tiled layer stickifies each K slice once.
struct StickCache {
  int64_t rows = -1;
  int64_t cols = -1;
  ZdnnTensor tensor;
};

struct ZdnnLinearHandler {
  zdnn_data_types type = FP32;
  int64_t k = 0;
  int64_t n = 0;
  size_t elem_size = 0;
  uint32_t dim1_limit = 0;
  uint32_t dim2_limit = 0;
  uint64_t mts = 0;
  int64_t k_tile = 0;
  int64_t n_tile = 0;
  std::vector<std::unique_ptr<MatmulTile>> tiles;
  std::vector<std::unique_ptr<StickCache>> activations;
  std::vector<std::unique_ptr<StickCache>> outputs;
  std::vector<uint8_t> act_scratch;
  int64_t cached_m = -1;
  int64_t cached_rows = -1;
  std::mutex mu;

  bool shape_fits(int64_t rows, int64_t k_len, int64_t n_len) const {
    if (rows <= 0 || k_len <= 0 || n_len <= 0) {
      return false;
    }
    const auto dim1 = static_cast<int64_t>(dim1_limit);
    const auto dim2 = static_cast<int64_t>(dim2_limit);
    // Weight is (K, N), activation is (rows, K), output is (rows, N).
    if (k_len > dim2 || n_len > dim1 || rows > dim2 || k_len > dim1 ||
        n_len > dim1) {
      return false;
    }
    const auto rows_u = static_cast<uint32_t>(rows);
    const auto k_u = static_cast<uint32_t>(k_len);
    const auto n_u = static_cast<uint32_t>(n_len);
    return transformed_fits(ZDNN_2D, type, k_u, n_u, mts) &&
           transformed_fits(ZDNN_1D, type, n_u, 1, mts) &&
           transformed_fits(ZDNN_2D, type, rows_u, k_u, mts) &&
           transformed_fits(ZDNN_2D, type, rows_u, n_u, mts);
  }

  int64_t largest(int64_t hi, auto fits) const {
    int64_t best = 0;
    int64_t lo = 1;
    while (lo <= hi) {
      const int64_t mid = lo + (hi - lo) / 2;
      if (fits(mid)) {
        best = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return best;
  }

  int64_t tile_rows(int64_t m) {
    if (m == cached_m) {
      return cached_rows;
    }
    const int64_t hi = std::min(m, static_cast<int64_t>(dim2_limit));
    cached_rows = largest(
        hi, [&](int64_t rows) { return shape_fits(rows, k_tile, n_tile); });
    cached_m = m;
    return cached_rows;
  }

  StickCache* activation_cache(int64_t rows, int64_t cols, const void* raw) {
    for (auto& cache : activations) {
      if (cache->rows == rows && cache->cols == cols) {
        cache->tensor.reload(raw, elem_size, static_cast<uint32_t>(rows),
                             static_cast<uint32_t>(cols), true);
        return cache.get();
      }
    }
    auto cache = std::make_unique<StickCache>();
    cache->rows = rows;
    cache->cols = cols;
    cache->tensor.load(ZDNN_2D, type, static_cast<uint32_t>(rows),
                       static_cast<uint32_t>(cols), raw, elem_size, true);
    auto* ptr = cache.get();
    activations.push_back(std::move(cache));
    return ptr;
  }

  StickCache* output_cache(int64_t rows, int64_t cols) {
    for (auto& cache : outputs) {
      if (cache->rows == rows && cache->cols == cols) {
        return cache.get();
      }
    }
    auto cache = std::make_unique<StickCache>();
    cache->rows = rows;
    cache->cols = cols;
    // Matmul writes a transformed output, so the buffer must already be in
    // stick layout. A zero transform does that once per shape.
    cache->tensor.load(ZDNN_2D, type, static_cast<uint32_t>(rows),
                       static_cast<uint32_t>(cols), nullptr, elem_size);
    auto* ptr = cache.get();
    outputs.push_back(std::move(cache));
    return ptr;
  }

  const uint8_t* activation_tile(const uint8_t* rows_in, int64_t rows,
                                 const MatmulTile& tile) {
    if (tile.k0 == 0 && tile.k_len == k) {
      return rows_in;
    }
    const size_t bytes =
        static_cast<size_t>(rows) * static_cast<size_t>(tile.k_len) * elem_size;
    act_scratch.resize(bytes);
    for (int64_t r = 0; r < rows; ++r) {
      const size_t src = (static_cast<size_t>(r) * static_cast<size_t>(k) +
                          static_cast<size_t>(tile.k0)) *
                         elem_size;
      const size_t dst =
          static_cast<size_t>(r) * static_cast<size_t>(tile.k_len) * elem_size;
      std::memcpy(act_scratch.data() + dst, rows_in + src,
                  static_cast<size_t>(tile.k_len) * elem_size);
    }
    return act_scratch.data();
  }

  void matmul(const StickCache* act, const MatmulTile& tile,
              StickCache* out_stick) {
    zdnn_check(zdnn_matmul_op(&act->tensor.z, &tile.weight.z, &tile.bias.z,
                              MATMUL_OP_ADDITION, &out_stick->tensor.z),
               "matmul_op");
  }

  void publish(StickCache* out_stick, int64_t rows, int64_t cols,
               const torch::Tensor& dest) {
    if (dest.is_contiguous()) {
      unstick(&out_stick->tensor.z, dest.data_ptr());
      return;
    }
    auto tmp = torch::empty({rows, cols}, dest.options());
    unstick(&out_stick->tensor.z, tmp.data_ptr());
    dest.copy_(tmp);
  }

  torch::Tensor take_fp32(StickCache* out_stick, int64_t rows, int64_t cols,
                          const torch::TensorOptions& options) {
    auto tmp = torch::empty({rows, cols}, options);
    unstick(&out_stick->tensor.z, tmp.data_ptr());
    return tmp.to(at::kFloat);
  }

  void run(torch::Tensor& out, const torch::Tensor& x) {
    std::lock_guard<std::mutex> lock(mu);
    TORCH_CHECK(x.dim() == 2, "zDNN linear expects a 2D activation");
    TORCH_CHECK(x.size(1) == k, "zDNN linear K mismatch");
    TORCH_CHECK(x.device().is_cpu() && out.device().is_cpu(),
                "zDNN linear tensors must be on CPU");
    TORCH_CHECK(host_type(x.scalar_type()) == type &&
                    x.scalar_type() == out.scalar_type(),
                "zDNN linear dtype mismatch");
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == n,
                "zDNN linear output shape mismatch");
    TORCH_CHECK(out.is_contiguous(), "zDNN linear output must be contiguous");
    const int64_t m = x.size(0);
    if (m == 0) {
      return;
    }
    const torch::Tensor x_c = x.contiguous();
    const int64_t row_tile = tile_rows(m);
    TORCH_CHECK(row_tile > 0, "activation does not fit NNPA tensor limits");

    int64_t k_groups = 0;
    for (const auto& tile : tiles) {
      if (tile->n0 == 0) {
        k_groups++;
      }
    }
    TORCH_CHECK(
        k_groups > 0 && tiles.size() % static_cast<size_t>(k_groups) == 0,
        "zDNN linear tile grid is not rectangular");
    const int64_t n_groups =
        static_cast<int64_t>(tiles.size() / static_cast<size_t>(k_groups));

    const auto* in_bytes = static_cast<const uint8_t*>(x_c.data_ptr());
    int64_t row = 0;
    while (row < m) {
      const int64_t rows = std::min(row_tile, m - row);
      const auto* row_in = in_bytes + static_cast<size_t>(row) *
                                          static_cast<size_t>(k) * elem_size;
      std::vector<torch::Tensor> k_acc;
      if (k_groups > 1) {
        k_acc.resize(static_cast<size_t>(n_groups));
      }
      for (int64_t ki = 0; ki < k_groups; ++ki) {
        const MatmulTile& proto = *tiles[static_cast<size_t>(ki)];
        const uint8_t* act_raw = activation_tile(row_in, rows, proto);
        StickCache* act = activation_cache(rows, proto.k_len, act_raw);
        for (int64_t ni = 0; ni < n_groups; ++ni) {
          const size_t index = static_cast<size_t>(ni * k_groups + ki);
          const MatmulTile& tile = *tiles[index];
          StickCache* out_stick = output_cache(rows, tile.n_len);
          matmul(act, tile, out_stick);
          auto dest = out.narrow(0, row, rows).narrow(1, tile.n0, tile.n_len);
          if (k_groups == 1) {
            publish(out_stick, rows, tile.n_len, dest);
            continue;
          }
          auto partial = take_fp32(out_stick, rows, tile.n_len, out.options());
          if (ki == 0) {
            k_acc[static_cast<size_t>(ni)] = std::move(partial);
          } else {
            k_acc[static_cast<size_t>(ni)].add_(partial);
          }
          if (ki + 1 == k_groups) {
            dest.copy_(k_acc[static_cast<size_t>(ni)]);
          }
        }
      }
      row += rows;
    }
  }
};

void pack_weight_tile(const uint8_t* weight, int64_t n, const MatmulTile& tile,
                      size_t elem_size, std::vector<uint8_t>* packed) {
  packed->resize(static_cast<size_t>(tile.k_len) *
                 static_cast<size_t>(tile.n_len) * elem_size);
  for (int64_t r = 0; r < tile.k_len; ++r) {
    const size_t src =
        (static_cast<size_t>(tile.k0 + r) * static_cast<size_t>(n) +
         static_cast<size_t>(tile.n0)) *
        elem_size;
    const size_t dst =
        static_cast<size_t>(r) * static_cast<size_t>(tile.n_len) * elem_size;
    std::memcpy(packed->data() + dst, weight + src,
                static_cast<size_t>(tile.n_len) * elem_size);
  }
}

}  // namespace

bool zdnn_linear_available() {
  static const bool available = []() {
    zdnn_init();
    return zdnn_is_nnpa_installed() &&
           zdnn_is_nnpa_function_installed(1, NNPA_MATMUL_OP);
  }();
  return available;
}

int64_t create_zdnn_linear_handler(const torch::Tensor& weight,
                                   const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(zdnn_linear_available(), "NNPA-MATMUL-OP is not installed");
  TORCH_CHECK(weight.dim() == 2, "zDNN weight must be 2D [K, N]");
  TORCH_CHECK(weight.device().is_cpu(), "zDNN weight must be on CPU");
  const torch::Tensor weight_c = weight.contiguous();
  const int64_t k = weight_c.size(0);
  const int64_t n = weight_c.size(1);
  TORCH_CHECK(k > 0 && n > 0, "zDNN weight dimensions must be positive");

  auto handler = std::make_unique<ZdnnLinearHandler>();
  handler->type = host_type(weight_c.scalar_type());
  handler->k = k;
  handler->n = n;
  handler->elem_size = host_elem_size(handler->type);
  const uint32_t mdis = zdnn_get_nnpa_max_dim_idx_size();
  handler->dim1_limit = dim_limit(1, mdis);
  handler->dim2_limit = dim_limit(2, mdis);
  handler->mts = zdnn_get_nnpa_max_tensor_size();
  TORCH_CHECK(handler->dim1_limit > 0 && handler->dim2_limit > 0,
              "NNPA maximum dimension index is zero");

  handler->k_tile = handler->largest(
      std::min(k, static_cast<int64_t>(handler->dim2_limit)),
      [&](int64_t k_len) { return handler->shape_fits(1, k_len, 1); });
  TORCH_CHECK(handler->k_tile > 0, "zDNN linear cannot fit a single K element");
  handler->n_tile =
      handler->largest(std::min(n, static_cast<int64_t>(handler->dim1_limit)),
                       [&](int64_t n_len) {
                         return handler->shape_fits(1, handler->k_tile, n_len);
                       });
  TORCH_CHECK(handler->n_tile > 0, "zDNN linear cannot fit a single N element");

  const auto* weight_bytes = static_cast<const uint8_t*>(weight_c.data_ptr());
  const uint8_t* bias_bytes = nullptr;
  torch::Tensor bias_c;
  std::vector<uint8_t> zero_bias;
  if (bias.has_value() && bias->defined() && bias->numel() > 0) {
    TORCH_CHECK(bias->device().is_cpu(), "zDNN bias must be on CPU");
    TORCH_CHECK(bias->dim() == 1 && bias->size(0) == n,
                "zDNN bias must have shape [N]");
    bias_c = bias->to(weight_c.dtype()).contiguous();
    bias_bytes = static_cast<const uint8_t*>(bias_c.data_ptr());
  } else {
    zero_bias.assign(static_cast<size_t>(n) * handler->elem_size, 0);
    bias_bytes = zero_bias.data();
  }

  std::vector<uint8_t> packed;
  std::vector<uint8_t> bias_tile;
  for (int64_t n0 = 0; n0 < n; n0 += handler->n_tile) {
    const int64_t n_len = std::min(handler->n_tile, n - n0);
    for (int64_t k0 = 0; k0 < k; k0 += handler->k_tile) {
      auto tile = std::make_unique<MatmulTile>();
      tile->k0 = k0;
      tile->k_len = std::min(handler->k_tile, k - k0);
      tile->n0 = n0;
      tile->n_len = n_len;
      pack_weight_tile(weight_bytes, n, *tile, handler->elem_size, &packed);
      tile->weight.load(ZDNN_2D, handler->type,
                        static_cast<uint32_t>(tile->k_len),
                        static_cast<uint32_t>(tile->n_len), packed.data(),
                        handler->elem_size);
      bias_tile.assign(static_cast<size_t>(n_len) * handler->elem_size, 0);
      // Bias is added on the first K tile. Later K tiles add a zero bias.
      if (k0 == 0) {
        std::memcpy(bias_tile.data(),
                    bias_bytes + static_cast<size_t>(n0) * handler->elem_size,
                    bias_tile.size());
      }
      tile->bias.load(ZDNN_1D, handler->type, static_cast<uint32_t>(n_len), 1,
                      bias_tile.data(), handler->elem_size);
      handler->tiles.push_back(std::move(tile));
    }
  }

  return reinterpret_cast<int64_t>(handler.release());
}

void release_zdnn_linear_handler(int64_t handler) {
  delete reinterpret_cast<ZdnnLinearHandler*>(handler);
}

void zdnn_linear(torch::Tensor& out, const torch::Tensor& x,
                 const torch::Tensor& handler_tensor) {
  auto* handler =
      reinterpret_cast<ZdnnLinearHandler*>(handler_tensor.item<int64_t>());
  TORCH_CHECK(handler != nullptr, "zDNN linear handler is null");
  handler->run(out, x);
}
