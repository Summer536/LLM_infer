#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "test_flashattention.h"

// kv cache shape = [numlayers, bs, kv head num, max_seq_len, head size]
// Flash-Decoding: 基于Flash-Decoding思路增加KV序列长度维度的并行性，提升GPU利用率
// 算法思路: 将KV cache分割成多个chunks → 并行计算每个chunk的注意力贡献 → 全局归约合并结果
// INT8量化相关常量定义
#define INT8_SCALE_FACTOR 127.0f
#define INT8_INV_SCALE_FACTOR (1.0f / 127.0f)

template<typename T>
__device__ T warpReduceSum(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31) / 32;
    static __shared__ T warpsum[64];
    
    val = warpReduceSum<T>(val);
    if (lane_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceSum<T>(warp_val);
}

template<typename T>
__device__ T warpReduceMax(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31) / 32;
    static __shared__ T warpmax[64];
    
    val = warpReduceMax(val);
    if (lane_id == 0){
        warpmax[warp_id] = val;
    }
    __syncthreads();
    
    T warp_val = tid < warp_nums ? warpmax[tid] : (T)0;
    return warpReduceMax(warp_val);
}

// Flash-Decoding 参数结构体
struct FlashDecodingParams {
    int num_chunks;
    int chunk_size;
    float* chunk_max;
    float* chunk_sum;
    float* chunk_output;
    
    __device__ FlashDecodingParams() : num_chunks(0), chunk_size(0), 
                                      chunk_max(nullptr), chunk_sum(nullptr), chunk_output(nullptr) {}
};

// 计算Q*K点积的特化函数
template<typename T>
__device__ float compute_qk_dot_product(typename Vec<T>::Type sq_vec, typename Vec<T>::Type kvec, float scale);

template<>
__device__ float compute_qk_dot_product<float>(float4 sq_vec, float4 kvec, float scale) {
    return (sq_vec.x * kvec.x + sq_vec.y * kvec.y + sq_vec.z * kvec.z + sq_vec.w * kvec.w) * scale;
}

template<>
__device__ float compute_qk_dot_product<half>(half2 sq_vec, half2 kvec, float scale) {
    return (__half2float(sq_vec.x) * __half2float(kvec.x) + 
            __half2float(sq_vec.y) * __half2float(kvec.y)) * scale;
}

template<>
__device__ float compute_qk_dot_product<int8_t>(char4 sq_vec, char4 kvec, float scale) {
    float q_vals[4] = {(float)sq_vec.x * INT8_INV_SCALE_FACTOR,
                      (float)sq_vec.y * INT8_INV_SCALE_FACTOR,
                      (float)sq_vec.z * INT8_INV_SCALE_FACTOR,
                      (float)sq_vec.w * INT8_INV_SCALE_FACTOR};
    float k_vals[4] = {(float)kvec.x * INT8_INV_SCALE_FACTOR,
                      (float)kvec.y * INT8_INV_SCALE_FACTOR,
                      (float)kvec.z * INT8_INV_SCALE_FACTOR,
                      (float)kvec.w * INT8_INV_SCALE_FACTOR};
    return (q_vals[0] * k_vals[0] + q_vals[1] * k_vals[1] + 
            q_vals[2] * k_vals[2] + q_vals[3] * k_vals[3]) * scale;
}

// 加权累加V向量的特化函数
template<typename T>
__device__ void accumulate_v_output(typename Vec<T>::Type& output, typename Vec<T>::Type vvec, float weight);

template<>
__device__ void accumulate_v_output<float>(float4& output, float4 vvec, float weight) {
    output.x += vvec.x * weight;
    output.y += vvec.y * weight;
    output.z += vvec.z * weight;
    output.w += vvec.w * weight;
}

template<>
__device__ void accumulate_v_output<half>(half2& output, half2 vvec, float weight) {
    half weight_h = __float2half(weight);
    output.x = __hfma(vvec.x, weight_h, output.x);
    output.y = __hfma(vvec.y, weight_h, output.y);
}

template<>
__device__ void accumulate_v_output<int8_t>(char4& output, char4 vvec, float weight) {
    // INT8: 在float域计算，避免精度损失
    float v_vals[4] = {(float)vvec.x * INT8_INV_SCALE_FACTOR,
                      (float)vvec.y * INT8_INV_SCALE_FACTOR,
                      (float)vvec.z * INT8_INV_SCALE_FACTOR,
                      (float)vvec.w * INT8_INV_SCALE_FACTOR};
    
    float out_vals[4] = {(float)output.x * INT8_INV_SCALE_FACTOR,
                        (float)output.y * INT8_INV_SCALE_FACTOR,
                        (float)output.z * INT8_INV_SCALE_FACTOR,
                        (float)output.w * INT8_INV_SCALE_FACTOR};
    
    out_vals[0] += v_vals[0] * weight;
    out_vals[1] += v_vals[1] * weight;
    out_vals[2] += v_vals[2] * weight;
    out_vals[3] += v_vals[3] * weight;
    
    output.x = (int8_t)__float2int_rn(out_vals[0] * INT8_SCALE_FACTOR);
    output.y = (int8_t)__float2int_rn(out_vals[1] * INT8_SCALE_FACTOR);
    output.z = (int8_t)__float2int_rn(out_vals[2] * INT8_SCALE_FACTOR);
    output.w = (int8_t)__float2int_rn(out_vals[3] * INT8_SCALE_FACTOR);
}

// 全局归约时的加权累加特化函数
template<typename T>
__device__ void accumulate_final_output(typename Vec<T>::Type& accumulated, typename Vec<T>::Type chunk_out, float correction);

template<>
__device__ void accumulate_final_output<float>(float4& accumulated, float4 chunk_out, float correction) {
    accumulated.x += chunk_out.x * correction;
    accumulated.y += chunk_out.y * correction;
    accumulated.z += chunk_out.z * correction;
    accumulated.w += chunk_out.w * correction;
}

template<>
__device__ void accumulate_final_output<half>(half2& accumulated, half2 chunk_out, float correction) {
    half correction_h = __float2half(correction);
    accumulated.x = __hfma(chunk_out.x, correction_h, accumulated.x);
    accumulated.y = __hfma(chunk_out.y, correction_h, accumulated.y);
}

template<>
__device__ void accumulate_final_output<int8_t>(char4& accumulated, char4 chunk_out, float correction) {
    float chunk_vals[4] = {(float)chunk_out.x * INT8_INV_SCALE_FACTOR,
                          (float)chunk_out.y * INT8_INV_SCALE_FACTOR,
                          (float)chunk_out.z * INT8_INV_SCALE_FACTOR,
                          (float)chunk_out.w * INT8_INV_SCALE_FACTOR};
    
    float acc_vals[4] = {(float)accumulated.x * INT8_INV_SCALE_FACTOR,
                        (float)accumulated.y * INT8_INV_SCALE_FACTOR,
                        (float)accumulated.z * INT8_INV_SCALE_FACTOR,
                        (float)accumulated.w * INT8_INV_SCALE_FACTOR};
    
    acc_vals[0] += chunk_vals[0] * correction;
    acc_vals[1] += chunk_vals[1] * correction;
    acc_vals[2] += chunk_vals[2] * correction;
    acc_vals[3] += chunk_vals[3] * correction;
    
    accumulated.x = (int8_t)__float2int_rn(acc_vals[0] * INT8_SCALE_FACTOR);
    accumulated.y = (int8_t)__float2int_rn(acc_vals[1] * INT8_SCALE_FACTOR);
    accumulated.z = (int8_t)__float2int_rn(acc_vals[2] * INT8_SCALE_FACTOR);
    accumulated.w = (int8_t)__float2int_rn(acc_vals[3] * INT8_SCALE_FACTOR);
}

// Flash-Decoding Forward Kernel
// 并行计算每个KV chunk的注意力贡献，增加KV序列长度维度的并行性
template<typename T>
__global__ void flash_decoding_forward_kernel(
    T* q,
    T* k_cache,
    T* v_cache,
    FlashDecodingParams params,
    T* chunk_outputs,
    const int batch_size,
    const int head_num,
    const int kv_head_num,
    const int max_seq_len,
    const int head_size,
    const int step,
    const int chunk_id
) {
    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    
    // GQA/MQA支持：多个Q head可能对应同一个KV head
    int kv_head_id = head_id / (head_num / kv_head_num);
    
    // 计算当前chunk的范围
    int chunk_start = chunk_id * params.chunk_size;
    int chunk_end = min(chunk_start + params.chunk_size, step);
    int actual_chunk_size = chunk_end - chunk_start;
    
    if (actual_chunk_size <= 0) return;
    
    int vec_size = Vec<T>::size;
    
    // 偏移量计算
    int q_batch_stride = head_num * head_size;
    int q_offset_vec = batch_id * q_batch_stride + head_id * head_size + tid * vec_size;
    
    int kv_batch_stride = kv_head_num * max_seq_len * head_size;
    int kv_head_stride = max_seq_len * head_size;
    int cache_base_offset = batch_id * kv_batch_stride + kv_head_id * kv_head_stride;
    
    // 加载Q向量到shared memory
    extern __shared__ char smem[];
    T* sq_scalar = reinterpret_cast<T*>(smem);
    float* chunk_logits = reinterpret_cast<float*>(sq_scalar + head_size);
    
    using Vec_t = typename Vec<T>::Type;
    Vec_t* sq = reinterpret_cast<Vec_t*>(sq_scalar);
    
    if (tid * vec_size < head_size) {
        sq[tid] = *reinterpret_cast<Vec_t*>(&q[q_offset_vec]);
    }
    __syncthreads();
    
    float scale = rsqrt(float(head_size));
    
    // 计算Q*K for current chunk (分块矩阵乘法)
    float chunk_max_val = -INFINITY;
    
    for (int pos = chunk_start; pos < chunk_end; pos++) {
        int k_offset = cache_base_offset + pos * head_size + tid * vec_size;
        
        Vec_t kvec = tid * vec_size < head_size ? 
                     *reinterpret_cast<Vec_t*>(&k_cache[k_offset]) : 
                     scalar_cast_vec<Vec_t, T>((T)0.0f);
        
        // 计算Q*K（向量化点积）
        float qk_score = 0.0f;
        if (tid * vec_size < head_size) {
            qk_score = compute_qk_dot_product<T>(sq[tid], kvec, scale);
        }
        
        float attn_score = blockReduceSum<float>(qk_score);
        
        if (tid == 0) {
            chunk_logits[pos - chunk_start] = attn_score;
            chunk_max_val = max(chunk_max_val, attn_score);
        }
        __syncthreads();
    }
    
    // 计算chunk级别的softmax统计信息
    float local_max = tid < actual_chunk_size ? chunk_logits[tid] : -INFINITY;
    float block_max = blockReduceMax<float>(local_max);
    
    __shared__ float shared_chunk_max;
    if (tid == 0) {
        shared_chunk_max = block_max;
    }
    __syncthreads();
    
    float local_sum = 0.0f;
    if (tid < actual_chunk_size) {
        local_sum = expf(chunk_logits[tid] - shared_chunk_max);
        chunk_logits[tid] = local_sum;
    }
    
    float block_sum = blockReduceSum<float>(local_sum);
    __shared__ float shared_chunk_sum;
    if (tid == 0) {
        shared_chunk_sum = block_sum;
    }
    __syncthreads();
    
    // 计算当前chunk的注意力输出 (Attention * V)
    if (tid * vec_size < head_size) {
        Vec_t chunk_output = scalar_cast_vec<Vec_t, T>((T)0.0f);
        
        for (int pos = chunk_start; pos < chunk_end; pos++) {
            int v_offset = cache_base_offset + pos * head_size + tid * vec_size;
            Vec_t vvec = *reinterpret_cast<Vec_t*>(&v_cache[v_offset]);
            float attn_weight = chunk_logits[pos - chunk_start];
            accumulate_v_output<T>(chunk_output, vvec, attn_weight);
        }
        
        // 存储chunk输出和统计信息
        int output_offset = ((batch_id * head_num + head_id) * params.num_chunks + chunk_id) * head_size + tid * vec_size;
        *reinterpret_cast<Vec_t*>(&chunk_outputs[output_offset]) = chunk_output;
        
        if (tid == 0) {
            int stats_offset = (batch_id * head_num + head_id) * params.num_chunks + chunk_id;
            params.chunk_max[stats_offset] = shared_chunk_max;
            params.chunk_sum[stats_offset] = shared_chunk_sum;
        }
    }
}

// Flash-Decoding Reduce Kernel
// 将所有chunks的结果合并成最终的注意力输出
template<typename T>
__global__ void flash_decoding_reduce_kernel(
    FlashDecodingParams params,
    T* chunk_outputs,
    T* final_output,
    const int batch_size,
    const int head_num,
    const int head_size
) {
    int tid = threadIdx.x;
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    
    if (tid * vec_size >= head_size) return;
    
    // 计算全局最大值
    int stats_base = (batch_id * head_num + head_id) * params.num_chunks;
    
    float global_max = -INFINITY;
    for (int i = 0; i < params.num_chunks; i++) {
        global_max = max(global_max, params.chunk_max[stats_base + i]);
    }
    
    // 计算全局归一化因子
    float global_sum = 0.0f;
    for (int i = 0; i < params.num_chunks; i++) {
        float chunk_max = params.chunk_max[stats_base + i];
        float chunk_sum = params.chunk_sum[stats_base + i];
        global_sum += chunk_sum * expf(chunk_max - global_max);
    }
    
    // 合并各chunk的输出
    Vec_t accumulated_output = scalar_cast_vec<Vec_t, T>((T)0.0f);
    
    for (int chunk_id = 0; chunk_id < params.num_chunks; chunk_id++) {
        float chunk_max = params.chunk_max[stats_base + chunk_id];
        float chunk_sum = params.chunk_sum[stats_base + chunk_id];
        float correction_factor = (chunk_sum * expf(chunk_max - global_max)) / global_sum;
        
        int chunk_output_offset = ((batch_id * head_num + head_id) * params.num_chunks + chunk_id) * head_size + tid * vec_size;
        Vec_t chunk_out = *reinterpret_cast<Vec_t*>(&chunk_outputs[chunk_output_offset]);
        
        accumulate_final_output<T>(accumulated_output, chunk_out, correction_factor);
    }
    
    // 写入最终结果
    int final_offset = batch_id * head_num * head_size + head_id * head_size + tid * vec_size;
    *reinterpret_cast<Vec_t*>(&final_output[final_offset]) = accumulated_output;
}

// 内存管理辅助函数
void allocate_flash_decoding_buffers(FlashDecodingParams& params, 
                                   int batch_size, int head_num, int num_chunks, int head_size) {
    int stats_size = batch_size * head_num * num_chunks;
    int output_size = batch_size * head_num * num_chunks * head_size;
    
    cudaMalloc(&params.chunk_max, stats_size * sizeof(float));
    cudaMalloc(&params.chunk_sum, stats_size * sizeof(float));
    cudaMalloc(&params.chunk_output, output_size * sizeof(float));
    
    params.num_chunks = num_chunks;
}

void free_flash_decoding_buffers(FlashDecodingParams& params) {
    if (params.chunk_max) cudaFree(params.chunk_max);
    if (params.chunk_sum) cudaFree(params.chunk_sum);
    if (params.chunk_output) cudaFree(params.chunk_output);
    
    params.chunk_max = nullptr;
    params.chunk_sum = nullptr;
    params.chunk_output = nullptr;
    params.num_chunks = 0;
}

int compute_optimal_chunk_size(int step, int head_size, int max_chunks, int available_smem) {
    // 基于shared memory限制计算最大chunk大小
    int smem_per_chunk = head_size * sizeof(float) + step * sizeof(float);
    int max_chunk_by_smem = available_smem / smem_per_chunk;
    
    // 确保有足够的并行度
    int min_chunks = min(max_chunks, 8);
    int chunk_size = max(1, (step + min_chunks - 1) / min_chunks);
    
    return min(chunk_size, max_chunk_by_smem);
}

template<typename T>
void launchFlashDecodingMHA(TensorWrapper<T>* qkv_buf,
                           BaseWeight<T>& qkv,
                           TensorWrapper<int>* layer_id,
                           TensorWrapper<T>* k_cache,
                           TensorWrapper<T>* v_cache,
                           TensorWrapper<bool>* finished,
                           TensorWrapper<int>* step,
                           TensorWrapper<T>* mha_output,
                           LLaMAAttentionStaticParams& static_params) {
    
    // qkv_buf shape = [bs, qkv_head_num, head_size]
    // k_cache shape = [num layers, bs, kv_head_num, max_seq_len, head_size]
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2];
    const int max_seq_len = k_cache->shape[3];
    const int head_size = qkv_buf->shape[2];
    const int head_num = qkv_head_num - 2 * kv_head_num;
    
    const int cur_step = step->getVal();
    const int layer = layer_id->getVal();
    
    if (cur_step <= 0) return;
    
    // 计算最优的chunk配置
    int available_smem = 48 * 1024;  // 48KB shared memory
    int max_chunks = min(16, (cur_step + 31) / 32);
    
    int optimal_chunk_size = compute_optimal_chunk_size(cur_step, head_size, max_chunks, available_smem);
    int num_chunks = (cur_step + optimal_chunk_size - 1) / optimal_chunk_size;
    
    // 内存分配
    FlashDecodingParams params;
    allocate_flash_decoding_buffers(params, batch_size, head_num, num_chunks, head_size);
    params.chunk_size = optimal_chunk_size;
    
    T* chunk_outputs;
    size_t chunk_output_size = batch_size * head_num * num_chunks * head_size * sizeof(T);
    cudaMalloc(&chunk_outputs, chunk_output_size);
    
    const int layer_offset = layer * max_seq_len * batch_size * kv_head_num * head_size;
    
    T* qkv_data = qkv_buf->data;
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;
    T* v = qkv_data + (head_num + kv_head_num) * head_size;
    
    // Phase 1: Forward kernels (并行处理各个chunks)
    dim3 forward_grid(batch_size, head_num, 1);
    dim3 forward_block(head_size);
    size_t forward_smem = head_size * sizeof(T) + optimal_chunk_size * sizeof(float);
    
    for (int chunk_id = 0; chunk_id < num_chunks; chunk_id++) {
        flash_decoding_forward_kernel<T><<<forward_grid, forward_block, forward_smem>>>(
            q,
            k_cache->data + layer_offset,
            v_cache->data + layer_offset,
            params,
            chunk_outputs,
            batch_size,
            head_num,
            kv_head_num,
            max_seq_len,
            head_size,
            cur_step,
            chunk_id
        );
    }
    
    cudaDeviceSynchronize();
    
    // Phase 2: Reduce kernel (合并所有chunks的结果)
    dim3 reduce_grid(batch_size, head_num);
    dim3 reduce_block(head_size);
    
    flash_decoding_reduce_kernel<T><<<reduce_grid, reduce_block>>>(
        params,
        chunk_outputs,
        mha_output->data,
        batch_size,
        head_num,
        head_size
    );
    
    // 内存清理
    cudaFree(chunk_outputs);
    free_flash_decoding_buffers(params);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Flash-Decoding MHA kernel error: %s\n", cudaGetErrorString(err));
    }
}

template void launchFlashDecodingMHA(TensorWrapper<float>* qkv_buf,
                                    BaseWeight<float>& qkv,
                                    TensorWrapper<int>* layer_id,
                                    TensorWrapper<float>* k_cache,
                                    TensorWrapper<float>* v_cache,
                                    TensorWrapper<bool>* finished,
                                    TensorWrapper<int>* step,
                                    TensorWrapper<float>* mha_output,
                                    LLaMAAttentionStaticParams& static_params);

template void launchFlashDecodingMHA(TensorWrapper<half>* qkv_buf,
                                   BaseWeight<half>& qkv,
                                   TensorWrapper<int>* layer_id,
                                   TensorWrapper<half>* k_cache,
                                   TensorWrapper<half>* v_cache,
                                   TensorWrapper<bool>* finished,
                                   TensorWrapper<int>* step,
                                   TensorWrapper<half>* mha_output,
                                   LLaMAAttentionStaticParams& static_params);

template void launchFlashDecodingMHA(TensorWrapper<int8_t>* qkv_buf,
                                    BaseWeight<int8_t>& qkv,
                                    TensorWrapper<int>* layer_id,
                                    TensorWrapper<int8_t>* k_cache,
                                    TensorWrapper<int8_t>* v_cache,
                                    TensorWrapper<bool>* finished,
                                    TensorWrapper<int>* step,
                                    TensorWrapper<int8_t>* mha_output,
                                    LLaMAAttentionStaticParams& static_params); 