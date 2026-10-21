// Linear layers on the IBM Z Neural Network Processing Assist (NNPA) via zDNN.
//
// Mirrors the sgl-kernels convert_weight_packed / weight_packed_linear split:
// the weight is transposed and stickified to DLFLOAT16 once at load time, so
// each forward only stickifies the activation, issues one
// NNPA-MATMUL-OP-BCAST23, and unstickifies the result.
//
// NNPA data-layout-format 0 stores a (E4, E3, E2, E1) tensor as
// [E4][ceil(E1 / 64)][E3][ceil(E2 / 32) * 32][64] DLFLOAT16 elements, with
// every 32 x 64 tile filling one 4K page. Two consequences are used below:
//   - each E4 slice is a contiguous block, so the activation is split into
//     stacked M-slices that are stickified/unstickified in parallel;
//   - for a 2D (K, N) tensor each group of 64 columns is a contiguous block,
//     so the weight can be packed in parallel and split along N when N
//     exceeds the maximum dimension index size.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include "cpu/micro_gemm/cpu_micro_gemm_zdnn.hpp"

// Some packaged zdnn.h versions lack their own extern "C" guard.
extern "C" {
#include <zdnn.h>
}

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {

constexpr int64_t kStickRows = 32;
constexpr int64_t kStickCols = 64;
constexpr int64_t kPageBytes = 4096;

#define ZDNN_CHECK(call)                                                       \
  do {                                                                         \
    const zdnn_status status_ = (call);                                        \
    TORCH_CHECK(status_ == ZDNN_OK || status_ == ZDNN_ELEMENT_RANGE_VIOLATION, \
                #call " failed: ", zdnn_get_status_message(status_));          \
  } while (0)

inline int64_t div_up(int64_t a, int64_t b) { return (a + b - 1) / b; }

// Bytes of a stickified (rows, cols) DLFLOAT16 matrix.
inline int64_t stick_bytes(int64_t rows, int64_t cols) {
  return div_up(rows, kStickRows) * div_up(cols, kStickCols) * kPageBytes;
}

struct NnpaLimits {
  int64_t max_dim;
  int64_t max_bytes;
};

const NnpaLimits& nnpa_limits() {
  static const NnpaLimits limits{
      static_cast<int64_t>(zdnn_get_nnpa_max_dim_idx_size()),
      static_cast<int64_t>(zdnn_get_nnpa_max_tensor_size())};
  return limits;
}

zdnn_data_types to_zdnn_type(at::ScalarType t) {
  switch (t) {
    case at::kBFloat16:
      return BFLOAT;
    case at::kHalf:
      return FP16;
    case at::kFloat:
      return FP32;
    default:
      TORCH_CHECK(false, "zDNN linear: unsupported dtype ", t);
  }
}

template <typename T>
bool all_in_dlf16_range(const T* x, int64_t n) {
  const float max = std::ldexp(1.0f + 510.0f / 512.0f, 32);
  for (int64_t i = 0; i < n; ++i) {
    if (!(std::abs(static_cast<float>(x[i])) <= max)) {
      return false;
    }
  }
  return true;
}

// zDNN fails to stickify NaN, Inf or out-of-range values, but has already
// written them as NINF, which the NNPA propagates to the output. Such inputs
// (e.g. uninitialized buffers in vLLM's profiling run) then give non-finite
// outputs like any GEMM, so only a failure on in-range data is an error.
void check_activation_stickify(zdnn_status status, const uint8_t* src,
                               int64_t n, zdnn_data_types type) {
  if (status == ZDNN_CONVERT_FAILURE) {
    bool in_range = true;
    switch (type) {
      case BFLOAT:
        in_range =
            all_in_dlf16_range(reinterpret_cast<const c10::BFloat16*>(src), n);
        break;
      case FP16:
        in_range =
            all_in_dlf16_range(reinterpret_cast<const c10::Half*>(src), n);
        break;
      default:
        in_range = all_in_dlf16_range(reinterpret_cast<const float*>(src), n);
    }
    if (!in_range) {
      TORCH_WARN_ONCE(
          "zDNN linear: input has NaN, Inf or values beyond the DLFLOAT16 "
          "range; the affected outputs are non-finite.");
      return;
    }
  }
  TORCH_CHECK(status == ZDNN_OK || status == ZDNN_ELEMENT_RANGE_VIOLATION,
              "zDNN linear: activation stickify failed: ",
              zdnn_get_status_message(status));
}

// A ztensor plus the descriptors it points to.
struct ZTensor {
  zdnn_tensor_desc pre_desc;
  zdnn_tensor_desc desc;
  zdnn_ztensor zt;

  template <typename... Dims>
  ZTensor(zdnn_data_layouts layout, zdnn_data_types type, void* buffer,
          bool is_transformed, Dims... dims) {
    zdnn_init_pre_transformed_desc(layout, type, &pre_desc,
                                   static_cast<uint32_t>(dims)...);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_desc, &desc));
    zdnn_init_ztensor(&pre_desc, &desc, &zt);
    zt.buffer_size = zdnn_getsize_ztensor(&desc);
    zt.buffer = buffer;
    zt.is_transformed = is_transformed;
  }
  ZTensor(const ZTensor&) = delete;
  ZTensor& operator=(const ZTensor&) = delete;
};

// Widest N chunk a single NNPA call can take for a (K, N) weight.
int64_t max_n_chunk(int64_t K) {
  const auto& limits = nnpa_limits();
  const int64_t by_dim = limits.max_dim / kStickCols * kStickCols;
  const int64_t by_size =
      limits.max_bytes / stick_bytes(K, kStickCols) * kStickCols;
  const int64_t n = std::min(by_dim, by_size);
  TORCH_CHECK(n >= kStickCols, "zDNN linear: K=", K, " exceeds NNPA limits");
  return n;
}

// Packed layout: stickified (K, N) weight followed by the stickified (N) bias.
int64_t packed_weight_bytes(int64_t N, int64_t K) {
  return stick_bytes(K, N) + stick_bytes(1, N);
}

uint8_t* aligned_workspace(int64_t bytes) {
  struct Free {
    void operator()(void* p) const { std::free(p); }
  };
  thread_local std::unique_ptr<void, Free> buffer;
  thread_local int64_t capacity = 0;
  if (bytes > capacity) {
    buffer.reset();
    buffer.reset(std::aligned_alloc(kPageBytes, bytes));
    TORCH_CHECK(buffer, "zDNN linear: failed to allocate ", bytes, " bytes");
    capacity = bytes;
  }
  return static_cast<uint8_t*>(buffer.get());
}

// Decode-sized M runs on the CPU cores instead of the NNPA: a matmul with few
// rows is bound by streaming the weight, which the NNPA does at well below
// memory bandwidth. All threads read the same stickified DLFLOAT16 weight.
constexpr int64_t kGemvMaxM = 8;
// Rows of a 64-column group swept at a time (32 KB of DLFLOAT16), so the four
// 16-column tiles reuse it from L1 and each line is fetched from memory once.
constexpr int64_t kGemvKBlock = 256;

// Splits K so the (group, split) work items divide evenly across threads;
// otherwise e.g. 40 groups on 16 threads leaves the slowest thread 3 groups.
int64_t gemv_k_splits(int64_t groups, int64_t K) {
  const int64_t threads = at::get_num_threads();
  int64_t best = 1;
  int64_t best_items = div_up(groups, threads);
  for (int64_t s = 2; s <= 4 && K / s >= kGemvKBlock; ++s) {
    const int64_t items = div_up(groups * s, threads);
    if (items * best < best_items * s) {
      best = s;
      best_items = items;
    }
  }
  return best;
}

// out[m, :cols] = c[m, :cols] + bias for a 64-column group.
template <typename scalar_t>
void store_group(scalar_t* out, int64_t ldo, float* c, int64_t ldc,
                 const uint16_t* bias, int64_t M, int64_t cols) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  constexpr int64_t tile_n = 16;
  for (int64_t t = 0; t < cols; t += tile_n) {
    const vec_op::FP32Vec16 b = cpu_micro_gemm::load_dlf16x16(bias + t);
    const int64_t nt = std::min(tile_n, cols - t);
    for (int64_t m = 0; m < M; ++m) {
      float* c_row = c + m * ldc + t;
      const vec_op::FP32Vec16 res = vec_op::FP32Vec16(c_row) + b;
      scalar_t* o = out + m * ldo + t;
      if (nt == tile_n) {
        scalar_vec_t(res).save(o);
      } else {
        res.save(c_row);
        for (int64_t j = 0; j < nt; ++j) {
          o[j] = static_cast<scalar_t>(c_row[j]);
        }
      }
    }
  }
}

template <typename scalar_t>
void gemv_packed_linear(scalar_t* out, int64_t ldo, const scalar_t* mat1,
                        int64_t lda, const uint8_t* w_base,
                        const uint8_t* b_base, int64_t M, int64_t N,
                        int64_t K) {
  using gemm_t = cpu_micro_gemm::TileGemmDLF16<float>;
  constexpr int64_t tile_n = gemm_t::NSize;
  const int64_t group_bytes = stick_bytes(K, kStickCols);
  const int64_t groups = div_up(N, kStickCols);
  const int64_t splits = gemv_k_splits(groups, K);
  const int64_t split_k = div_up(K, splits);
  const int64_t ldp = groups * kStickCols;

  // Widen the activation once so the micro-kernel splats it straight from
  // memory instead of converting inside the K loop.
  thread_local std::vector<float> a_fp32;
  thread_local std::vector<float> partial;
  a_fp32.resize(M * K);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t k = 0; k < K; ++k) {
      a_fp32[m * K + k] = static_cast<float>(mat1[m * lda + k]);
    }
  }
  const float* a = a_fp32.data();
  if (splits > 1) {
    partial.resize(splits * M * ldp);
  }
  float* p = partial.data();

  auto bias_of = [&](int64_t g) {
    return reinterpret_cast<const uint16_t*>(b_base + g * kPageBytes);
  };

  at::parallel_for(0, groups * splits, 1, [&](int64_t begin, int64_t end) {
    alignas(64) float c[kGemvMaxM * kStickCols];
    for (int64_t item = begin; item < end; ++item) {
      const int64_t g = item / splits;
      const int64_t s = item % splits;
      const int64_t k0 = s * split_k;
      const int64_t k1 = std::min(K, k0 + split_k);
      const int64_t cols = std::min(kStickCols, N - g * kStickCols);
      const auto* wg =
          reinterpret_cast<const uint16_t*>(w_base + g * group_bytes);
      for (int64_t kb = k0; kb < k1; kb += kGemvKBlock) {
        const int32_t kn = std::min(kGemvKBlock, k1 - kb);
        for (int64_t t = 0; t < cols; t += tile_n) {
          for (int64_t m0 = 0; m0 < M; m0 += gemm_t::MaxMSize) {
            gemm_t::gemm(a + m0 * K + kb, wg + kb * kStickCols + t,
                         c + m0 * kStickCols + t,
                         std::min<int64_t>(gemm_t::MaxMSize, M - m0), kn, K,
                         kStickCols, kb != k0);
          }
        }
      }
      if (splits == 1) {
        store_group(out + g * kStickCols, ldo, c, kStickCols, bias_of(g), M,
                    cols);
      } else {
        for (int64_t m = 0; m < M; ++m) {
          std::memcpy(p + (s * M + m) * ldp + g * kStickCols,
                      c + m * kStickCols, cols * sizeof(float));
        }
      }
    }
  });

  if (splits == 1) {
    return;
  }
  at::parallel_for(0, groups, 1, [&](int64_t begin, int64_t end) {
    for (int64_t g = begin; g < end; ++g) {
      const int64_t cols = std::min(kStickCols, N - g * kStickCols);
      float* c = p + g * kStickCols;
      for (int64_t s = 1; s < splits; ++s) {
        for (int64_t m = 0; m < M; ++m) {
          const float* src = c + (s * M + m) * ldp;
          for (int64_t j = 0; j < cols; ++j) {
            c[m * ldp + j] += src[j];
          }
        }
      }
      store_group(out + g * kStickCols, ldo, c, ldp, bias_of(g), M, cols);
    }
  });
}

}  // namespace

bool zdnn_is_available() { return zdnn_is_nnpa_installed(); }

// weight : [N, K]
// bias   : [N]
// return : uint8 [packed_weight_bytes(N, K)], 4K aligned
at::Tensor zdnn_convert_weight_packed(const at::Tensor& weight,
                                      const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(zdnn_is_nnpa_installed(), "zDNN linear: NNPA is not available");
  TORCH_CHECK(weight.dim() == 2, "zDNN linear: weight must be 2D");
  const int64_t N = weight.size(0);
  const int64_t K = weight.size(1);
  TORCH_CHECK(K <= nnpa_limits().max_dim, "zDNN linear: K=", K,
              " exceeds the NNPA maximum dimension index size ",
              nnpa_limits().max_dim);
  const zdnn_data_types type = to_zdnn_type(weight.scalar_type());
  const int64_t n_chunk = max_n_chunk(K);

  const at::Tensor b = bias.has_value()
                           ? bias->to(weight.scalar_type()).contiguous()
                           : at::zeros({N}, weight.options());
  TORCH_CHECK(b.numel() == N, "zDNN linear: bias must have N elements");

  const int64_t total = packed_weight_bytes(N, K);
  void* raw = std::aligned_alloc(kPageBytes, total);
  TORCH_CHECK(raw, "zDNN linear: failed to allocate ", total, " bytes");
  at::Tensor packed = at::from_blob(
      raw, {total}, [](void* p) { std::free(p); },
      at::TensorOptions().dtype(at::kByte));
  uint8_t* w_base = packed.data_ptr<uint8_t>();
  uint8_t* b_base = w_base + stick_bytes(K, N);

  const int64_t col_groups = div_up(N, kStickCols);
  const int64_t groups_per_chunk = n_chunk / kStickCols;
  at::parallel_for(0, col_groups, 1, [&](int64_t begin, int64_t end) {
    for (int64_t g = begin; g < end; g += groups_per_chunk) {
      const int64_t n0 = g * kStickCols;
      const int64_t n = std::min({n_chunk, (end - g) * kStickCols, N - n0});
      const at::Tensor w = weight.narrow(0, n0, n).t().contiguous();
      ZTensor zw(ZDNN_2D, type, w_base + g * stick_bytes(K, kStickCols), false,
                 K, n);
      TORCH_CHECK(static_cast<int64_t>(zw.zt.buffer_size) == stick_bytes(K, n),
                  "zDNN linear: unexpected stickified weight size");
      ZDNN_CHECK(zdnn_transform_ztensor(&zw.zt, w.data_ptr()));

      const at::Tensor bn = b.narrow(0, n0, n).contiguous();
      ZTensor zb(ZDNN_1D, type, b_base + g * kPageBytes, false, n);
      ZDNN_CHECK(zdnn_transform_ztensor(&zb.zt, bn.data_ptr()));
    }
  });
  return packed;
}

// out    : [M, N]
// mat1   : [M, K]
// packed : zdnn_convert_weight_packed(weight [N, K], bias)
void zdnn_weight_packed_linear(at::Tensor& out, const at::Tensor& mat1,
                               const at::Tensor& packed, int64_t N) {
  TORCH_CHECK(mat1.dim() == 2 && out.dim() == 2,
              "zDNN linear: mat1 and out must be 2D");
  TORCH_CHECK(mat1.stride(-1) == 1 && out.stride(-1) == 1,
              "zDNN linear: last dim of mat1 and out must be contiguous");
  TORCH_CHECK(out.scalar_type() == mat1.scalar_type(),
              "zDNN linear: out and mat1 must share a dtype");
  const int64_t M = mat1.size(0);
  const int64_t K = mat1.size(1);
  TORCH_CHECK(out.size(0) == M && out.size(1) == N,
              "zDNN linear: out must be [M, N]");
  TORCH_CHECK(packed.numel() == packed_weight_bytes(N, K),
              "zDNN linear: packed weight does not match N=", N, ", K=", K);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % kPageBytes == 0,
              "zDNN linear: packed weight must be 4K aligned");
  if (M == 0) {
    return;
  }

  uint8_t* w_base = static_cast<uint8_t*>(packed.data_ptr());
  uint8_t* b_base = w_base + stick_bytes(K, N);
  if (M <= kGemvMaxM) {
    AT_DISPATCH_SWITCH(
        mat1.scalar_type(), "zdnn_gemv",
        AT_DISPATCH_CASE_REDUCED_FLOATING_TYPES([&] {
          gemv_packed_linear<scalar_t>(out.data_ptr<scalar_t>(), out.stride(0),
                                       mat1.data_ptr<scalar_t>(),
                                       mat1.stride(0), w_base, b_base, M, N, K);
        }) AT_DISPATCH_CASE(at::kFloat, [&] {
          gemv_packed_linear<scalar_t>(out.data_ptr<scalar_t>(), out.stride(0),
                                       mat1.data_ptr<scalar_t>(),
                                       mat1.stride(0), w_base, b_base, M, N, K);
        }));
    return;
  }

  const zdnn_data_types type = to_zdnn_type(mat1.scalar_type());
  const auto& limits = nnpa_limits();
  const int64_t n_chunk = std::min(max_n_chunk(K), N);

  // Slice M into stacked row blocks: one per thread, each a multiple of the
  // 32-row stick height and within the NNPA dimension/size limits.
  const int64_t max_rows =
      std::min(limits.max_dim, limits.max_bytes /
                                   std::max(stick_bytes(kStickRows, K),
                                            stick_bytes(kStickRows, n_chunk)) *
                                   kStickRows) /
      kStickRows * kStickRows;
  TORCH_CHECK(max_rows >= kStickRows, "zDNN linear: shape exceeds NNPA limits");
  const int64_t rows =
      std::min(max_rows, div_up(div_up(M, at::get_num_threads()), kStickRows) *
                             kStickRows);
  const int64_t num_slices = div_up(M, rows);
  const int64_t a_slice_bytes = stick_bytes(rows, K);
  const int64_t slices_per_call = std::min(
      {num_slices, limits.max_dim,
       limits.max_bytes / std::max(a_slice_bytes, stick_bytes(rows, n_chunk))});

  const int64_t a_bytes = slices_per_call * a_slice_bytes;
  uint8_t* a_buf =
      aligned_workspace(a_bytes + slices_per_call * stick_bytes(rows, n_chunk));
  uint8_t* o_buf = a_buf + a_bytes;
  const int64_t elem = mat1.element_size();
  const uint8_t* x_ptr = static_cast<const uint8_t*>(mat1.data_ptr());
  const int64_t x_row_bytes = mat1.stride(0) * elem;
  uint8_t* out_ptr = static_cast<uint8_t*>(out.data_ptr());
  const int64_t out_row_bytes = out.stride(0) * elem;

  for (int64_t s0 = 0; s0 < num_slices; s0 += slices_per_call) {
    const int64_t S = std::min(slices_per_call, num_slices - s0);

    // Worker threads do not inherit InferenceMode, so ATen ops on the
    // caller's tensors are avoided inside parallel regions.
    at::parallel_for(0, S, 1, [&](int64_t begin, int64_t end) {
      std::vector<uint8_t> padded;
      for (int64_t s = begin; s < end; ++s) {
        const int64_t r0 = (s0 + s) * rows;
        const int64_t r = std::min(rows, M - r0);
        const uint8_t* src = x_ptr + r0 * x_row_bytes;
        if (r < rows || x_row_bytes != K * elem) {
          padded.assign(rows * K * elem, 0);
          for (int64_t i = 0; i < r; ++i) {
            std::memcpy(padded.data() + i * K * elem, src + i * x_row_bytes,
                        K * elem);
          }
          src = padded.data();
        }
        ZTensor za(ZDNN_2D, type, a_buf + s * a_slice_bytes, false, rows, K);
        check_activation_stickify(
            zdnn_transform_ztensor(&za.zt, const_cast<uint8_t*>(src)), src,
            rows * K, type);
      }
    });

    ZTensor za(ZDNN_3DS, type, a_buf, true, S, rows, K);
    for (int64_t n0 = 0; n0 < N; n0 += n_chunk) {
      const int64_t n = std::min(n_chunk, N - n0);
      const int64_t g = n0 / kStickCols;
      ZTensor zw(ZDNN_2D, type, w_base + g * stick_bytes(K, kStickCols), true,
                 K, n);
      ZTensor zb(ZDNN_1D, type, b_base + g * kPageBytes, true, n);
      ZTensor zo(ZDNN_3DS, type, o_buf, false, S, rows, n);
      ZDNN_CHECK(zdnn_matmul_bcast_op(&za.zt, &zw.zt, &zb.zt,
                                      MATMUL_BCAST_OP_ADDITION, &zo.zt));

      const int64_t o_slice_bytes = stick_bytes(rows, n);
      at::parallel_for(0, S, 1, [&](int64_t begin, int64_t end) {
        std::vector<uint8_t> dst;
        for (int64_t s = begin; s < end; ++s) {
          const int64_t r0 = (s0 + s) * rows;
          const int64_t r = std::min(rows, M - r0);
          uint8_t* out_rows = out_ptr + r0 * out_row_bytes + n0 * elem;
          ZTensor zs(ZDNN_2D, type, o_buf + s * o_slice_bytes, true, rows, n);
          if (r == rows && n * elem == out_row_bytes) {
            ZDNN_CHECK(zdnn_transform_origtensor(&zs.zt, out_rows));
          } else {
            dst.resize(rows * n * elem);
            ZDNN_CHECK(zdnn_transform_origtensor(&zs.zt, dst.data()));
            for (int64_t i = 0; i < r; ++i) {
              std::memcpy(out_rows + i * out_row_bytes,
                          dst.data() + i * n * elem, n * elem);
            }
          }
        }
      });
    }
  }
}
