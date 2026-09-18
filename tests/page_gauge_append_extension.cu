#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>

namespace {

constexpr int kHeadDim = 128;
constexpr int kPageSize = 16;
constexpr int kThreads = 128;
// The tail kernel launches one CTA per KV head (eight CTAs for Mistral-7B).
// Use a wider CTA there so the 256-token reductions expose enough independent
// work on SM120; the append kernels retain their one-thread-per-dimension CTA.
constexpr int kTailThreads = 512;
constexpr int kQueryGroupSize = 4;
constexpr int kThreadsPerQueryGroup = kTailThreads / kQueryGroupSize;

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    at::ScalarType dtype,
    int64_t dimensions) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.dim() == dimensions, name, " has the wrong rank");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_head_major_input(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be FP16");
  TORCH_CHECK(tensor.dim() == 3, name, " must have rank three");
  TORCH_CHECK(tensor.stride(2) == 1 && tensor.stride(1) == kHeadDim,
              name, " must be head-major with a contiguous head dimension");
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float block_max(float value) {
  __shared__ float warp_values[4];
  __shared__ float result;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_max(value);
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float aggregate = lane < 4 ? warp_values[lane] : 0.0f;
    aggregate = warp_max(aggregate);
    if (lane == 0) result = aggregate;
  }
  __syncthreads();
  return result;
}

__device__ __forceinline__ float block_max_signed(float value) {
  __shared__ float warp_values[4];
  __shared__ float result;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_max(value);
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float aggregate = lane < 4 ? warp_values[lane] : -FLT_MAX;
    aggregate = warp_max(aggregate);
    if (lane == 0) result = aggregate;
  }
  __syncthreads();
  return result;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum(float value) {
  __shared__ float warp_values[4];
  __shared__ float result;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();
  if (warp == 0) {
    float aggregate = lane < 4 ? warp_values[lane] : 0.0f;
    aggregate = warp_sum(aggregate);
    if (lane == 0) result = aggregate;
  }
  __syncthreads();
  return result;
}

// Four independent 128-thread reductions execute concurrently inside the
// 512-thread tail CTA.  Every thread must call these helpers because they use
// block-wide barriers, while each result remains private to one GQA row.
__device__ __forceinline__ float query_group_max_signed(float value, int group) {
  __shared__ float warp_values[kQueryGroupSize][4];
  __shared__ float result[kQueryGroupSize];
  const int group_thread = threadIdx.x % kThreadsPerQueryGroup;
  const int lane = group_thread & 31;
  const int group_warp = group_thread >> 5;
  value = warp_max(value);
  if (lane == 0) warp_values[group][group_warp] = value;
  __syncthreads();
  if (group_warp == 0) {
    float aggregate = lane < 4 ? warp_values[group][lane] : -FLT_MAX;
    aggregate = warp_max(aggregate);
    if (lane == 0) result[group] = aggregate;
  }
  __syncthreads();
  return result[group];
}

__device__ __forceinline__ float query_group_sum(float value, int group) {
  __shared__ float warp_values[kQueryGroupSize][4];
  __shared__ float result[kQueryGroupSize];
  const int group_thread = threadIdx.x % kThreadsPerQueryGroup;
  const int lane = group_thread & 31;
  const int group_warp = group_thread >> 5;
  value = warp_sum(value);
  if (lane == 0) warp_values[group][group_warp] = value;
  __syncthreads();
  if (group_warp == 0) {
    float aggregate = lane < 4 ? warp_values[group][lane] : 0.0f;
    aggregate = warp_sum(aggregate);
    if (lane == 0) result[group] = aggregate;
  }
  __syncthreads();
  return result[group];
}

__device__ __forceinline__ float rotate_element(
    const __half* input,
    const __half* cosine,
    const __half* sine,
    int head,
    int dimension) {
  const int paired = dimension < kHeadDim / 2
      ? dimension + kHeadDim / 2
      : dimension - kHeadDim / 2;
  unsigned short partner = __half_as_ushort(input[head * kHeadDim + paired]);
  if (dimension < kHeadDim / 2) partner ^= 0x8000u;
  // Match HF apply_rotary_pos_emb exactly: both products and the following
  // addition materialize in the FP16 tensor dtype.  A float FMA is
  // algebraically equivalent but stores different generated K values, whose
  // tiny per-layer differences can amplify over a long recurrent decode.
  const unsigned short primary =
      __half_as_ushort(input[head * kHeadDim + dimension]);
  const unsigned short cosine_value = __half_as_ushort(cosine[dimension]);
  const unsigned short sine_value = __half_as_ushort(sine[dimension]);
  unsigned short primary_product;
  unsigned short partner_product;
  unsigned short result;
  asm volatile("mul.rn.f16 %0, %1, %2;"
               : "=h"(primary_product)
               : "h"(primary), "h"(cosine_value));
  asm volatile("mul.rn.f16 %0, %1, %2;"
               : "=h"(partner_product)
               : "h"(partner), "h"(sine_value));
  asm volatile("add.rn.f16 %0, %1, %2;"
               : "=h"(result)
               : "h"(primary_product), "h"(partner_product));
  return __half2float(__ushort_as_half(result));
}

__global__ void rope_append_fp16_kernel(
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    const __half* __restrict__ cosine,
    const __half* __restrict__ sine,
    __half* __restrict__ rotated_query,
    __half* __restrict__ key_pages,
    __half* __restrict__ value_pages,
    size_t query_batch_stride,
    size_t key_batch_stride,
    size_t value_batch_stride,
    int batch_size,
    int query_heads,
    int kv_heads,
    int pages_per_request,
    int host_position,
    const int32_t* __restrict__ device_position,
    size_t rope_row_stride,
    int rope_rows) {
  const int position =
      device_position == nullptr ? host_position : device_position[0];
  if (position < 0 || position >= pages_per_request * kPageSize ||
      (device_position != nullptr && position >= rope_rows)) {
    return;
  }
  if (device_position != nullptr) {
    cosine += static_cast<size_t>(position) * rope_row_stride;
    sine += static_cast<size_t>(position) * rope_row_stride;
  }
  const int dimension = threadIdx.x;
  const int blocks_per_request = query_heads + kv_heads;
  const int request = static_cast<int>(blockIdx.x) / blocks_per_request;
  if (request >= batch_size) return;
  const int request_block = static_cast<int>(blockIdx.x) % blocks_per_request;
  const int page = position / kPageSize;
  const int token = position % kPageSize;
  const __half* request_query =
      query + static_cast<size_t>(request) * query_batch_stride;
  const __half* request_key =
      key + static_cast<size_t>(request) * key_batch_stride;
  const __half* request_value =
      value + static_cast<size_t>(request) * value_batch_stride;
  __half* request_rotated_query =
      rotated_query + static_cast<size_t>(request) * query_heads * kHeadDim;
  if (request_block < query_heads) {
    const int head = request_block;
    request_rotated_query[head * kHeadDim + dimension] = __float2half_rn(
        rotate_element(request_query, cosine, sine, head, dimension));
    return;
  }
  const int head = request_block - query_heads;
  if (head >= kv_heads) return;
  const int physical_page = request * pages_per_request + page;
  const size_t destination =
      (((size_t)physical_page * kPageSize + token) * kv_heads + head) * kHeadDim +
      dimension;
  key_pages[destination] = __float2half_rn(
      rotate_element(request_key, cosine, sine, head, dimension));
  value_pages[destination] = request_value[head * kHeadDim + dimension];
}

__global__ void rope_append_page_gauge_kernel(
    const __half* __restrict__ query,
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    const __half* __restrict__ cosine,
    const __half* __restrict__ sine,
    const __half* __restrict__ key_center,
    const __half* __restrict__ value_center,
    __half* __restrict__ rotated_query,
    __half* __restrict__ exact_key_pages,
    __half* __restrict__ exact_value_pages,
    int8_t* __restrict__ key_codes,
    int8_t* __restrict__ value_codes,
    __half* __restrict__ key_scales,
    __half* __restrict__ value_scales,
    size_t query_batch_stride,
    size_t key_batch_stride,
    size_t value_batch_stride,
    int batch_size,
    int query_heads,
    int kv_heads,
    int exact_pages_per_request,
    int exact_sink_pages,
    int code_pages_per_request,
    int host_position,
    const int32_t* __restrict__ device_position,
    size_t rope_row_stride,
    int rope_rows) {
  const int position =
      device_position == nullptr ? host_position : device_position[0];
  if (position < 0 || position >= code_pages_per_request * kPageSize ||
      (device_position != nullptr && position >= rope_rows)) {
    return;
  }
  if (device_position != nullptr) {
    cosine += static_cast<size_t>(position) * rope_row_stride;
    sine += static_cast<size_t>(position) * rope_row_stride;
  }
  const int dimension = threadIdx.x;
  const int blocks_per_request = query_heads + kv_heads;
  const int request = static_cast<int>(blockIdx.x) / blocks_per_request;
  if (request >= batch_size) return;
  const int request_block = static_cast<int>(blockIdx.x) % blocks_per_request;
  const int page = position / kPageSize;
  const int exact_tail_pages = exact_pages_per_request - exact_sink_pages;
  const int exact_local_page = page < exact_sink_pages
      ? page
      : exact_sink_pages + page % exact_tail_pages;
  const int exact_page =
      request * exact_pages_per_request + exact_local_page;
  const int code_page = request * code_pages_per_request + page;
  const int token = position % kPageSize;
  const __half* request_query =
      query + static_cast<size_t>(request) * query_batch_stride;
  const __half* request_key =
      key + static_cast<size_t>(request) * key_batch_stride;
  const __half* request_value =
      value + static_cast<size_t>(request) * value_batch_stride;
  __half* request_rotated_query =
      rotated_query + static_cast<size_t>(request) * query_heads * kHeadDim;
  if (request_block < query_heads) {
    const int head = request_block;
    request_rotated_query[head * kHeadDim + dimension] = __float2half_rn(
        rotate_element(request_query, cosine, sine, head, dimension));
    return;
  }
  const int head = request_block - query_heads;
  if (head >= kv_heads) return;
  const size_t destination =
      (((size_t)exact_page * kPageSize + token) * kv_heads + head) * kHeadDim +
      dimension;
  const int gauge_index = head * kHeadDim + dimension;
  const size_t center_request_offset =
      static_cast<size_t>(request) * kv_heads * kHeadDim;
  const float centered_key =
      rotate_element(request_key, cosine, sine, head, dimension) -
      __half2float(key_center[center_request_offset + gauge_index]);
  const float centered_value =
      __half2float(request_value[gauge_index]) -
      __half2float(value_center[center_request_offset + gauge_index]);
  exact_key_pages[destination] = __float2half_rn(centered_key);
  exact_value_pages[destination] = __float2half_rn(centered_value);
  __syncthreads();

  // The page stays exact for the recent-token window, but its compressed copy
  // is produced when the final token arrives.  It is therefore ready without
  // another launch when the page later exits that window.
  if (token != kPageSize - 1) return;

  float local_key_max = 0.0f;
  float local_value_max = 0.0f;
#pragma unroll
  for (int page_token = 0; page_token < kPageSize; ++page_token) {
    const size_t offset =
        (((size_t)exact_page * kPageSize + page_token) * kv_heads + head) *
            kHeadDim +
        dimension;
    local_key_max = fmaxf(
        local_key_max, fabsf(__half2float(exact_key_pages[offset])));
    local_value_max = fmaxf(
        local_value_max, fabsf(__half2float(exact_value_pages[offset])));
  }
  const float key_maximum = block_max(local_key_max);
  const float value_maximum = block_max(local_value_max);
  const __half stored_key_scale = __float2half_rn(
      fmaxf(key_maximum / 127.0f, exp2f(-20.0f)));
  const __half stored_value_scale = __float2half_rn(
      fmaxf(value_maximum / 127.0f, exp2f(-20.0f)));
  if (dimension == 0) {
    key_scales[code_page * kv_heads + head] = stored_key_scale;
    value_scales[code_page * kv_heads + head] = stored_value_scale;
  }
  const float key_scale = __half2float(stored_key_scale);
  const float value_scale = __half2float(stored_value_scale);
#pragma unroll
  for (int page_token = 0; page_token < kPageSize; ++page_token) {
    const size_t exact_offset =
        (((size_t)exact_page * kPageSize + page_token) * kv_heads + head) *
            kHeadDim +
        dimension;
    const size_t code_offset =
        (((size_t)code_page * kPageSize + page_token) * kv_heads + head) *
            kHeadDim +
        dimension;
    const float key_quotient =
        __half2float(exact_key_pages[exact_offset]) / key_scale;
    const float value_quotient =
        __half2float(exact_value_pages[exact_offset]) / value_scale;
    int quantized_key = key_quotient >= 0.0f
        ? static_cast<int>(floorf(key_quotient + 0.5f))
        : -static_cast<int>(floorf(-key_quotient + 0.5f));
    int quantized_value = value_quotient >= 0.0f
        ? static_cast<int>(floorf(value_quotient + 0.5f))
        : -static_cast<int>(floorf(-value_quotient + 0.5f));
    quantized_key = max(-127, min(127, quantized_key));
    quantized_value = max(-127, min(127, quantized_value));
    key_codes[code_offset] = static_cast<int8_t>(quantized_key);
    value_codes[code_offset] = static_cast<int8_t>(quantized_value);
  }
}

// Compute the short exact FP16 tail, merge it with FlashInfer's normalized
// old-cache state, and restore the PageGauge value center in one launch.
// FlashInfer exposes log-sum-exp in base 2, so tail scores use sm_scale*log2(e)
// and can be combined without converting the old state.
__global__ void exact_tail_merge_center_kernel(
    const __half* __restrict__ query,
    const __half* __restrict__ exact_key_pages,
    const __half* __restrict__ exact_value_pages,
    const int32_t* __restrict__ exact_page_indices,
    const int32_t* __restrict__ exact_last_page_len,
    __half* __restrict__ old_output,
    float* __restrict__ old_lse,
    const __half* __restrict__ value_center,
    int query_heads,
    int kv_heads,
    int num_exact_pages_per_request,
    float sm_scale_log2) {
  constexpr int kMaxExactTokens = 256;
  constexpr int kTailWarps = kTailThreads / 32;
  constexpr int kOutputVectors = kHeadDim / 2;
  constexpr int kValuePartitions = kTailThreads / kOutputVectors;
  static_assert(kTailThreads % kOutputVectors == 0);
  __shared__ float tail_weights[kQueryGroupSize][kMaxExactTokens];
  __shared__ float merged_maximum[kQueryGroupSize];
  __shared__ float old_weight[kQueryGroupSize];
  __shared__ float denominator[kQueryGroupSize];
  __shared__ float2
      tail_value_partials[kQueryGroupSize][kValuePartitions][kOutputVectors];

  const int request = static_cast<int>(blockIdx.x) / kv_heads;
  const int kv_head = static_cast<int>(blockIdx.x) % kv_heads;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int tail_tokens =
      (num_exact_pages_per_request - 1) * kPageSize +
      exact_last_page_len[request];
  const size_t query_request_offset =
      static_cast<size_t>(request) * query_heads * kHeadDim;
  const size_t lse_request_offset =
      static_cast<size_t>(request) * query_heads;

  const int query_dimension = lane * 4;
  float2 q_fragment_01[kQueryGroupSize];
  float2 q_fragment_23[kQueryGroupSize];
#pragma unroll
  for (int group = 0; group < kQueryGroupSize; ++group) {
    const int query_head = kv_head * kQueryGroupSize + group;
    const __half2* query_vector = reinterpret_cast<const __half2*>(
        query + query_request_offset + query_head * kHeadDim + query_dimension);
    q_fragment_01[group] = __half22float2(query_vector[0]);
    q_fragment_23[group] = __half22float2(query_vector[1]);
  }

  // One warp owns a token at a time. Its lanes load K once, then form all four
  // grouped-query scores before moving to the next token.  Sixteen warps cut
  // the longest per-warp chain from 64 tokens to 16 at a full tail.
  for (int token = warp; token < tail_tokens; token += kTailWarps) {
    const int logical_page = token / kPageSize;
    const int entry = token % kPageSize;
    const int physical_page = exact_page_indices[
        request * num_exact_pages_per_request + logical_page];
    const size_t key_base =
        (((size_t)physical_page * kPageSize + entry) * kv_heads + kv_head) *
        kHeadDim;
    const __half2* key_vector = reinterpret_cast<const __half2*>(
        exact_key_pages + key_base + query_dimension);
    const float2 key_fragment_01 = __half22float2(key_vector[0]);
    const float2 key_fragment_23 = __half22float2(key_vector[1]);
#pragma unroll
    for (int group = 0; group < kQueryGroupSize; ++group) {
      float score = fmaf(q_fragment_01[group].x, key_fragment_01.x, 0.0f);
      score = fmaf(q_fragment_01[group].y, key_fragment_01.y, score);
      score = fmaf(q_fragment_23[group].x, key_fragment_23.x, score);
      score = fmaf(q_fragment_23[group].y, key_fragment_23.y, score);
      score = warp_sum(score);
      if (lane == 0) {
        tail_weights[group][token] = score * sm_scale_log2;
      }
    }
  }
  __syncthreads();

  // Assign one 128-thread cohort to each GQA row, so the four online-softmax
  // normalizations run concurrently rather than serially in one CTA.
  const int softmax_group = threadIdx.x / kThreadsPerQueryGroup;
  const int group_thread = threadIdx.x % kThreadsPerQueryGroup;
  float local_max = -FLT_MAX;
  for (int token = group_thread; token < tail_tokens;
       token += kThreadsPerQueryGroup) {
    local_max = fmaxf(local_max, tail_weights[softmax_group][token]);
  }
  const float tail_max = query_group_max_signed(local_max, softmax_group);
  const int softmax_query_head =
      kv_head * kQueryGroupSize + softmax_group;
  if (group_thread == 0) {
    merged_maximum[softmax_group] =
        fmaxf(old_lse[lse_request_offset + softmax_query_head], tail_max);
    old_weight[softmax_group] =
        exp2f(
            old_lse[lse_request_offset + softmax_query_head] -
            merged_maximum[softmax_group]);
  }
  __syncthreads();

  float local_sum = 0.0f;
  for (int token = group_thread; token < tail_tokens;
       token += kThreadsPerQueryGroup) {
    const float weight = exp2f(
        tail_weights[softmax_group][token] -
        merged_maximum[softmax_group]);
    tail_weights[softmax_group][token] = weight;
    local_sum += weight;
  }
  const float tail_sum = query_group_sum(local_sum, softmax_group);
  if (group_thread == 0) {
    denominator[softmax_group] = old_weight[softmax_group] + tail_sum;
  }
  __syncthreads();

  // Eight threads split each half2 output vector over the tail.  This reduces
  // the full-tail dependency chain from 256 scalar iterations to 32 vector
  // iterations while loading each V element only once for all four GQA rows.
  const int output_vector = threadIdx.x % kOutputVectors;
  const int value_partition = threadIdx.x / kOutputVectors;
  const int dimension = output_vector * 2;
  float2 tail_value[kQueryGroupSize];
#pragma unroll
  for (int group = 0; group < kQueryGroupSize; ++group) {
    tail_value[group] = make_float2(0.0f, 0.0f);
  }
  for (int token = value_partition; token < tail_tokens;
       token += kValuePartitions) {
    const int logical_page = token / kPageSize;
    const int entry = token % kPageSize;
    const int physical_page = exact_page_indices[
        request * num_exact_pages_per_request + logical_page];
    const size_t value_offset =
        (((size_t)physical_page * kPageSize + entry) * kv_heads + kv_head) *
            kHeadDim +
        dimension;
    const float2 value = __half22float2(
        *reinterpret_cast<const __half2*>(exact_value_pages + value_offset));
#pragma unroll
    for (int group = 0; group < kQueryGroupSize; ++group) {
      tail_value[group].x = fmaf(
          tail_weights[group][token], value.x, tail_value[group].x);
      tail_value[group].y = fmaf(
          tail_weights[group][token], value.y, tail_value[group].y);
    }
  }
#pragma unroll
  for (int group = 0; group < kQueryGroupSize; ++group) {
    tail_value_partials[group][value_partition][output_vector] =
        tail_value[group];
  }
  __syncthreads();

  if (threadIdx.x < kOutputVectors) {
    const size_t center_request_offset =
        static_cast<size_t>(request) * kv_heads * kHeadDim;
    const float2 center = __half22float2(*reinterpret_cast<const __half2*>(
        value_center + center_request_offset + kv_head * kHeadDim + dimension));
#pragma unroll
    for (int group = 0; group < kQueryGroupSize; ++group) {
      float2 reduced_value = make_float2(0.0f, 0.0f);
#pragma unroll
      for (int partition = 0; partition < kValuePartitions; ++partition) {
        const float2 partial =
            tail_value_partials[group][partition][output_vector];
        reduced_value.x += partial.x;
        reduced_value.y += partial.y;
      }
      const int query_head = kv_head * kQueryGroupSize + group;
      const size_t output_offset =
          query_request_offset + static_cast<size_t>(query_head) * kHeadDim +
          dimension;
      const float2 old_value = __half22float2(
          *reinterpret_cast<const __half2*>(old_output + output_offset));
      const float denominator_reciprocal = __frcp_rn(denominator[group]);
      const float2 merged_value = make_float2(
          (old_weight[group] * old_value.x + reduced_value.x) *
                  denominator_reciprocal +
              center.x,
          (old_weight[group] * old_value.y + reduced_value.y) *
                  denominator_reciprocal +
              center.y);
      *reinterpret_cast<__half2*>(old_output + output_offset) =
          __floats2half2_rn(merged_value.x, merged_value.y);
    }
  }
  if (threadIdx.x == 0) {
#pragma unroll
    for (int group = 0; group < kQueryGroupSize; ++group) {
      const int query_head = kv_head * kQueryGroupSize + group;
      old_lse[lse_request_offset + query_head] =
          log2f(denominator[group]) + merged_maximum[group];
    }
  }
}

void check_common(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& cosine,
    const torch::Tensor& sine,
    const torch::Tensor& rotated_query,
    const torch::Tensor& key_pages,
    const torch::Tensor& value_pages,
    int64_t position) {
  check_head_major_input(query, "query");
  check_head_major_input(key, "key");
  check_head_major_input(value, "value");
  check_tensor(cosine, "cosine", at::kHalf, 1);
  check_tensor(sine, "sine", at::kHalf, 1);
  check_tensor(rotated_query, "rotated_query", at::kHalf, 3);
  check_tensor(key_pages, "key_pages", at::kHalf, 4);
  check_tensor(value_pages, "value_pages", at::kHalf, 4);
  TORCH_CHECK(query.size(0) > 0 && query.size(0) == key.size(0) &&
                  query.size(0) == value.size(0),
              "Q/K/V batch sizes must match and be positive");
  TORCH_CHECK(query.size(2) == kHeadDim && key.size(2) == kHeadDim &&
                  value.size(2) == kHeadDim,
              "head dimension must be 128");
  TORCH_CHECK(key.sizes() == value.sizes(), "K/V input shapes must match");
  TORCH_CHECK(rotated_query.sizes() == query.sizes(),
              "rotated query shape mismatch");
  TORCH_CHECK(cosine.numel() == kHeadDim && sine.numel() == kHeadDim,
              "RoPE vectors must contain 128 values");
  TORCH_CHECK(key_pages.sizes() == value_pages.sizes(),
              "K/V page shapes must match");
  TORCH_CHECK(key_pages.size(1) == kPageSize &&
                  key_pages.size(2) == key.size(1) &&
                  key_pages.size(3) == kHeadDim,
              "page cache must be NHD [pages,16,Hkv,128]");
  TORCH_CHECK(position >= 0, "position must be non-negative");
}

void check_dynamic_position_and_rope(
    const torch::Tensor& query,
    const torch::Tensor& cosine,
    const torch::Tensor& sine,
    const torch::Tensor& position) {
  check_tensor(cosine, "cosine", at::kHalf, 2);
  check_tensor(sine, "sine", at::kHalf, 2);
  check_tensor(position, "position", at::kInt, 1);
  TORCH_CHECK(cosine.sizes() == sine.sizes() && cosine.size(0) > 0 &&
                  cosine.size(1) == kHeadDim,
              "dynamic RoPE tables must be matching [positions,128] tensors");
  TORCH_CHECK(position.numel() == 1,
              "dynamic position must contain exactly one int32 scalar");
  TORCH_CHECK(cosine.get_device() == query.get_device() &&
                  sine.get_device() == query.get_device() &&
                  position.get_device() == query.get_device(),
              "dynamic position, RoPE tables, and Q/K/V must share one device");
}

}  // namespace

void rope_append_fp16(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cosine,
    torch::Tensor sine,
    torch::Tensor rotated_query,
    torch::Tensor key_pages,
    torch::Tensor value_pages,
    int64_t position) {
  check_common(query, key, value, cosine, sine, rotated_query, key_pages,
               value_pages, position);
  TORCH_CHECK(position / kPageSize < key_pages.size(0),
              "position exceeds the allocated FP16 cache");
  const int batch_size = static_cast<int>(query.size(0));
  TORCH_CHECK(key_pages.size(0) % batch_size == 0,
              "FP16 cache pages must divide evenly across requests");
  const int pages_per_request =
      static_cast<int>(key_pages.size(0) / batch_size);
  TORCH_CHECK(position / kPageSize < pages_per_request,
              "position exceeds the per-request FP16 cache capacity");
  const int query_heads = static_cast<int>(query.size(1));
  const int kv_heads = static_cast<int>(key.size(1));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rope_append_fp16_kernel<<<
      batch_size * (query_heads + kv_heads), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(cosine.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(sine.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(rotated_query.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(key_pages.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(value_pages.data_ptr<at::Half>()),
      static_cast<size_t>(query.stride(0)),
      static_cast<size_t>(key.stride(0)),
      static_cast<size_t>(value.stride(0)),
      batch_size, query_heads, kv_heads, pages_per_request,
      static_cast<int>(position), nullptr, 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_append_fp16_dynamic(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cosine,
    torch::Tensor sine,
    torch::Tensor position,
    torch::Tensor rotated_query,
    torch::Tensor key_pages,
    torch::Tensor value_pages) {
  // Reuse the complete Q/K/V/cache validation without reading the device
  // position back to the host. The actual row and destination are selected by
  // the kernel from the persistent int32 scalar captured by CUDA Graph.
  check_head_major_input(query, "query");
  check_head_major_input(key, "key");
  check_head_major_input(value, "value");
  check_tensor(rotated_query, "rotated_query", at::kHalf, 3);
  check_tensor(key_pages, "key_pages", at::kHalf, 4);
  check_tensor(value_pages, "value_pages", at::kHalf, 4);
  TORCH_CHECK(query.size(0) > 0 && query.size(0) == key.size(0) &&
                  query.size(0) == value.size(0),
              "Q/K/V batch sizes must match and be positive");
  TORCH_CHECK(query.size(2) == kHeadDim && key.size(2) == kHeadDim &&
                  value.size(2) == kHeadDim,
              "head dimension must be 128");
  TORCH_CHECK(key.sizes() == value.sizes(), "K/V input shapes must match");
  TORCH_CHECK(rotated_query.sizes() == query.sizes(),
              "rotated query shape mismatch");
  TORCH_CHECK(key_pages.sizes() == value_pages.sizes() &&
                  key_pages.size(1) == kPageSize &&
                  key_pages.size(2) == key.size(1) &&
                  key_pages.size(3) == kHeadDim,
              "page cache must be NHD [pages,16,Hkv,128]");
  check_dynamic_position_and_rope(query, cosine, sine, position);
  const int batch_size = static_cast<int>(query.size(0));
  TORCH_CHECK(key_pages.size(0) % batch_size == 0,
              "FP16 cache pages must divide evenly across requests");
  const int pages_per_request =
      static_cast<int>(key_pages.size(0) / batch_size);
  const int query_heads = static_cast<int>(query.size(1));
  const int kv_heads = static_cast<int>(key.size(1));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rope_append_fp16_kernel<<<
      batch_size * (query_heads + kv_heads), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(cosine.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(sine.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(rotated_query.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(key_pages.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(value_pages.data_ptr<at::Half>()),
      static_cast<size_t>(query.stride(0)),
      static_cast<size_t>(key.stride(0)),
      static_cast<size_t>(value.stride(0)),
      batch_size, query_heads, kv_heads, pages_per_request, 0,
      position.data_ptr<int32_t>(), static_cast<size_t>(cosine.stride(0)),
      static_cast<int>(cosine.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_append_page_gauge(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cosine,
    torch::Tensor sine,
    torch::Tensor key_center,
    torch::Tensor value_center,
    torch::Tensor rotated_query,
    torch::Tensor exact_key_pages,
    torch::Tensor exact_value_pages,
    torch::Tensor key_codes,
    torch::Tensor value_codes,
    torch::Tensor key_scales,
    torch::Tensor value_scales,
    int64_t exact_sink_pages,
    int64_t position) {
  check_common(query, key, value, cosine, sine, rotated_query, exact_key_pages,
               exact_value_pages, position);
  TORCH_CHECK(key_center.is_cuda() && value_center.is_cuda(),
              "centers must be CUDA");
  TORCH_CHECK(key_center.scalar_type() == at::kHalf &&
                  value_center.scalar_type() == at::kHalf,
              "centers must be FP16");
  TORCH_CHECK(key_center.is_contiguous() && value_center.is_contiguous(),
              "centers must be contiguous");
  check_tensor(key_codes, "key_codes", at::kChar, 4);
  check_tensor(value_codes, "value_codes", at::kChar, 4);
  check_tensor(key_scales, "key_scales", at::kHalf, 2);
  check_tensor(value_scales, "value_scales", at::kHalf, 2);
  const int query_heads = static_cast<int>(query.size(1));
  const int kv_heads = static_cast<int>(key.size(1));
  const int batch_size = static_cast<int>(query.size(0));
  TORCH_CHECK(
      key_center.sizes() == value_center.sizes() &&
          ((batch_size == 1 && key_center.dim() == 2 &&
            key_center.size(0) == kv_heads &&
            key_center.size(1) == kHeadDim) ||
           (key_center.dim() == 3 && key_center.size(0) == batch_size &&
            key_center.size(1) == kv_heads &&
            key_center.size(2) == kHeadDim)),
      "centers must be [B,Hkv,128] (or [Hkv,128] for batch one)");
  TORCH_CHECK(key_codes.sizes() == value_codes.sizes() &&
                  key_codes.size(1) == kPageSize &&
                  key_codes.size(2) == kv_heads &&
                  key_codes.size(3) == kHeadDim,
              "INT8 caches must be NHD [pages,16,Hkv,128]");
  TORCH_CHECK(exact_key_pages.size(0) > 0 &&
                  exact_key_pages.size(0) % batch_size == 0 &&
                  key_codes.size(0) % batch_size == 0,
              "PageGauge pages must be positive and divide evenly across requests");
  const int exact_pages_per_request =
      static_cast<int>(exact_key_pages.size(0) / batch_size);
  TORCH_CHECK(exact_sink_pages >= 0,
              "exact prefix page count must be nonnegative");
  TORCH_CHECK(exact_pages_per_request > exact_sink_pages,
              "exact cache must retain at least one tail-ring page");
  const int code_pages_per_request =
      static_cast<int>(key_codes.size(0) / batch_size);
  TORCH_CHECK(exact_sink_pages < code_pages_per_request,
              "exact prefix exceeds the logical cache capacity");
  TORCH_CHECK(position / kPageSize < code_pages_per_request,
              "position exceeds the per-request PageGauge cache capacity");
  TORCH_CHECK(key_scales.sizes() == value_scales.sizes() &&
                  key_scales.size(0) == key_codes.size(0) &&
                  key_scales.size(1) == kv_heads,
              "scales must be [pages,Hkv]");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rope_append_page_gauge_kernel<<<
      batch_size * (query_heads + kv_heads), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(cosine.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(sine.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key_center.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value_center.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(rotated_query.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(exact_key_pages.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(exact_value_pages.data_ptr<at::Half>()),
      key_codes.data_ptr<int8_t>(), value_codes.data_ptr<int8_t>(),
      reinterpret_cast<__half*>(key_scales.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(value_scales.data_ptr<at::Half>()),
      static_cast<size_t>(query.stride(0)),
      static_cast<size_t>(key.stride(0)),
      static_cast<size_t>(value.stride(0)),
      batch_size, query_heads, kv_heads, exact_pages_per_request,
      static_cast<int>(exact_sink_pages),
      code_pages_per_request, static_cast<int>(position), nullptr, 0, 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rope_append_page_gauge_dynamic(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor cosine,
    torch::Tensor sine,
    torch::Tensor position,
    torch::Tensor key_center,
    torch::Tensor value_center,
    torch::Tensor rotated_query,
    torch::Tensor exact_key_pages,
    torch::Tensor exact_value_pages,
    torch::Tensor key_codes,
    torch::Tensor value_codes,
    torch::Tensor key_scales,
    torch::Tensor value_scales,
    int64_t exact_sink_pages) {
  check_head_major_input(query, "query");
  check_head_major_input(key, "key");
  check_head_major_input(value, "value");
  check_tensor(rotated_query, "rotated_query", at::kHalf, 3);
  check_tensor(exact_key_pages, "exact_key_pages", at::kHalf, 4);
  check_tensor(exact_value_pages, "exact_value_pages", at::kHalf, 4);
  TORCH_CHECK(query.size(0) > 0 && query.size(0) == key.size(0) &&
                  query.size(0) == value.size(0),
              "Q/K/V batch sizes must match and be positive");
  TORCH_CHECK(query.size(2) == kHeadDim && key.size(2) == kHeadDim &&
                  value.size(2) == kHeadDim,
              "head dimension must be 128");
  TORCH_CHECK(key.sizes() == value.sizes(), "K/V input shapes must match");
  TORCH_CHECK(rotated_query.sizes() == query.sizes(),
              "rotated query shape mismatch");
  TORCH_CHECK(exact_key_pages.sizes() == exact_value_pages.sizes() &&
                  exact_key_pages.size(1) == kPageSize &&
                  exact_key_pages.size(2) == key.size(1) &&
                  exact_key_pages.size(3) == kHeadDim,
              "exact page cache must be NHD [pages,16,Hkv,128]");
  check_dynamic_position_and_rope(query, cosine, sine, position);
  TORCH_CHECK(key_center.is_cuda() && value_center.is_cuda(),
              "centers must be CUDA");
  TORCH_CHECK(key_center.scalar_type() == at::kHalf &&
                  value_center.scalar_type() == at::kHalf,
              "centers must be FP16");
  TORCH_CHECK(key_center.is_contiguous() && value_center.is_contiguous(),
              "centers must be contiguous");
  check_tensor(key_codes, "key_codes", at::kChar, 4);
  check_tensor(value_codes, "value_codes", at::kChar, 4);
  check_tensor(key_scales, "key_scales", at::kHalf, 2);
  check_tensor(value_scales, "value_scales", at::kHalf, 2);
  const int query_heads = static_cast<int>(query.size(1));
  const int kv_heads = static_cast<int>(key.size(1));
  const int batch_size = static_cast<int>(query.size(0));
  TORCH_CHECK(
      key_center.sizes() == value_center.sizes() &&
          ((batch_size == 1 && key_center.dim() == 2 &&
            key_center.size(0) == kv_heads &&
            key_center.size(1) == kHeadDim) ||
           (key_center.dim() == 3 && key_center.size(0) == batch_size &&
            key_center.size(1) == kv_heads &&
            key_center.size(2) == kHeadDim)),
      "centers must be [B,Hkv,128] (or [Hkv,128] for batch one)");
  TORCH_CHECK(key_codes.sizes() == value_codes.sizes() &&
                  key_codes.size(1) == kPageSize &&
                  key_codes.size(2) == kv_heads &&
                  key_codes.size(3) == kHeadDim,
              "INT8 caches must be NHD [pages,16,Hkv,128]");
  TORCH_CHECK(exact_key_pages.size(0) > 0 &&
                  exact_key_pages.size(0) % batch_size == 0 &&
                  key_codes.size(0) % batch_size == 0,
              "PageGauge pages must be positive and divide evenly across requests");
  const int exact_pages_per_request =
      static_cast<int>(exact_key_pages.size(0) / batch_size);
  TORCH_CHECK(exact_sink_pages >= 0,
              "exact prefix page count must be nonnegative");
  TORCH_CHECK(exact_pages_per_request > exact_sink_pages,
              "exact cache must retain at least one tail-ring page");
  const int code_pages_per_request =
      static_cast<int>(key_codes.size(0) / batch_size);
  TORCH_CHECK(exact_sink_pages < code_pages_per_request,
              "exact prefix exceeds the logical cache capacity");
  TORCH_CHECK(key_scales.sizes() == value_scales.sizes() &&
                  key_scales.size(0) == key_codes.size(0) &&
                  key_scales.size(1) == kv_heads,
              "scales must be [pages,Hkv]");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rope_append_page_gauge_kernel<<<
      batch_size * (query_heads + kv_heads), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(cosine.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(sine.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key_center.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value_center.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(rotated_query.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(exact_key_pages.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(exact_value_pages.data_ptr<at::Half>()),
      key_codes.data_ptr<int8_t>(), value_codes.data_ptr<int8_t>(),
      reinterpret_cast<__half*>(key_scales.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(value_scales.data_ptr<at::Half>()),
      static_cast<size_t>(query.stride(0)),
      static_cast<size_t>(key.stride(0)),
      static_cast<size_t>(value.stride(0)),
      batch_size, query_heads, kv_heads, exact_pages_per_request,
      static_cast<int>(exact_sink_pages),
      code_pages_per_request, 0, position.data_ptr<int32_t>(),
      static_cast<size_t>(cosine.stride(0)),
      static_cast<int>(cosine.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void exact_tail_merge_center(
    torch::Tensor query,
    torch::Tensor exact_key_pages,
    torch::Tensor exact_value_pages,
    torch::Tensor exact_page_indices,
    torch::Tensor exact_last_page_len,
    torch::Tensor old_output,
    torch::Tensor old_lse,
    torch::Tensor value_center,
    double sm_scale) {
  check_tensor(query, "query", at::kHalf, 3);
  check_tensor(exact_key_pages, "exact_key_pages", at::kHalf, 4);
  check_tensor(exact_value_pages, "exact_value_pages", at::kHalf, 4);
  check_tensor(exact_page_indices, "exact_page_indices", at::kInt, 1);
  check_tensor(exact_last_page_len, "exact_last_page_len", at::kInt, 1);
  check_tensor(old_output, "old_output", at::kHalf, 3);
  check_tensor(old_lse, "old_lse", at::kFloat, 2);
  TORCH_CHECK(value_center.is_cuda() &&
                  value_center.scalar_type() == at::kHalf &&
                  value_center.is_contiguous(),
              "value_center must be contiguous CUDA FP16");
  TORCH_CHECK(query.sizes() == old_output.sizes(),
              "query and old output shapes must match");
  TORCH_CHECK(query.size(2) == kHeadDim,
              "head dimension must be 128");
  TORCH_CHECK(exact_key_pages.sizes() == exact_value_pages.sizes() &&
                  exact_key_pages.size(1) == kPageSize &&
                  exact_key_pages.size(3) == kHeadDim,
              "exact caches must be matching NHD [pages,16,Hkv,128] tensors");
  const int query_heads = static_cast<int>(query.size(1));
  const int batch_size = static_cast<int>(query.size(0));
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  const int kv_heads = static_cast<int>(exact_key_pages.size(2));
  TORCH_CHECK(query_heads > 0 && kv_heads > 0 && query_heads % kv_heads == 0,
              "query heads must be a positive multiple of KV heads");
  TORCH_CHECK(query_heads / kv_heads == 4,
              "the fused exact-tail kernel currently requires GQA group size 4");
  TORCH_CHECK(old_lse.size(0) == batch_size &&
                  old_lse.size(1) == query_heads,
              "old_lse must be [B,Hq]");
  TORCH_CHECK(
      (batch_size == 1 && value_center.dim() == 2 &&
       value_center.size(0) == kv_heads &&
       value_center.size(1) == kHeadDim) ||
          (value_center.dim() == 3 && value_center.size(0) == batch_size &&
           value_center.size(1) == kv_heads &&
           value_center.size(2) == kHeadDim),
      "value_center must be [B,Hkv,128] (or [Hkv,128] for batch one)");
  TORCH_CHECK(exact_last_page_len.numel() == batch_size,
              "exact_last_page_len must contain one device scalar per request");
  TORCH_CHECK(exact_page_indices.numel() % batch_size == 0,
              "exact page indices must divide evenly across requests");
  const int num_exact_pages_per_request =
      static_cast<int>(exact_page_indices.numel() / batch_size);
  TORCH_CHECK(num_exact_pages_per_request > 0 &&
                  num_exact_pages_per_request <= 16,
              "the fused exact tail supports 1..16 pages per request");
  TORCH_CHECK(exact_key_pages.size(0) >=
                  batch_size * num_exact_pages_per_request,
              "exact page index count exceeds the ring allocation");
  TORCH_CHECK(sm_scale > 0.0 && std::isfinite(sm_scale),
              "sm_scale must be positive and finite");

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr float kLog2E = 1.4426950408889634074f;
  exact_tail_merge_center_kernel<<<
      batch_size * kv_heads, kTailThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(exact_key_pages.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(exact_value_pages.data_ptr<at::Half>()),
      exact_page_indices.data_ptr<int32_t>(),
      exact_last_page_len.data_ptr<int32_t>(),
      reinterpret_cast<__half*>(old_output.data_ptr<at::Half>()),
      old_lse.data_ptr<float>(),
      reinterpret_cast<const __half*>(value_center.data_ptr<at::Half>()),
      query_heads, kv_heads, num_exact_pages_per_request,
      static_cast<float>(sm_scale) * kLog2E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rope_append_fp16", &rope_append_fp16,
             "Fused RoPE and FP16 NHD cache append");
  module.def("rope_append_fp16_dynamic", &rope_append_fp16_dynamic,
             "Fused device-position RoPE and FP16 NHD cache append");
  module.def("rope_append_page_gauge", &rope_append_page_gauge,
             "Fused RoPE, centered exact append, and completed-page INT8 finalization");
  module.def("rope_append_page_gauge_dynamic", &rope_append_page_gauge_dynamic,
             "Fused device-position RoPE, centered exact append, and INT8 finalization");
  module.def("exact_tail_merge_center", &exact_tail_merge_center,
             "Fused exact-tail attention, online-state merge, and center restore");
}
