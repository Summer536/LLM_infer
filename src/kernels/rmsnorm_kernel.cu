#include "src/kernels/rmsnorm_kernel.h"
#include "src/utils/cuda_debug_utils.cuh"
#include <stdio.h>

template <typename T>
__device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val;
}

template <typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int laneid = tid % 32;
    int warpnum = (blockDim.x + 31) / 32;

    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    if(laneid == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    
    T sum = tid < warpnum ? warpsum[tid] : (T)0;

    sum = warpReduceSum<T>(sum);
    return sum;   
}

// output = input * 1 / sqrtf(mean(sigma(x^2)) + eps) * scale
template <typename T>
__global__ void RMSNorm(T *decoder_out,
                        T *decoder_residual,
                        T *scale,
                        float eps,
                        int num_tokens,
                        int hidden_units){
    using Vec_t = typename Vec<T>::Type;

    float thread_sum = 0.0f;
    Vec_t *dout = reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t*>(decoder_residual + blockIdx.x * hidden_units);

    for(int idx = threadIdx.x; idx < hidden_units / Vec<T>::size; idx += blockDim.x){
        Vec_t vec = dout[idx];
        rsd[idx] = vec;

        // For different data types, we need to accumulate in float precision
        if constexpr (std::is_same_v<T, int8_t>) {
            // INT8: dequantize to float for computation
            float val_x = static_cast<float>(vec.x) * INT8_INV_SCALE_FACTOR;
            float val_y = static_cast<float>(vec.y) * INT8_INV_SCALE_FACTOR;
            float val_z = static_cast<float>(vec.z) * INT8_INV_SCALE_FACTOR;
            float val_w = static_cast<float>(vec.w) * INT8_INV_SCALE_FACTOR;
            thread_sum += val_x * val_x;
            thread_sum += val_y * val_y;
            thread_sum += val_z * val_z;
            thread_sum += val_w * val_w;
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16: convert to float for computation
            float val_x = __half2float(vec.x);
            float val_y = __half2float(vec.y);
            thread_sum += val_x * val_x;
            thread_sum += val_y * val_y;
        } else {
            // FP32: direct computation
            thread_sum += vec.x * vec.x;
            thread_sum += vec.y * vec.y;
            thread_sum += vec.z * vec.z;
            thread_sum += vec.w * vec.w;
        }
    }

    thread_sum = blockReduceSum<float>(thread_sum);

    __shared__ float inv_mean;
    if(threadIdx.x == 0){
        inv_mean = rsqrtf((float)thread_sum / hidden_units + eps);
    }
    __syncthreads();

    Vec_t *s = reinterpret_cast<Vec_t*>(scale);
    for(int idx = threadIdx.x; idx < hidden_units / Vec<T>::size; idx += blockDim.x){
        Vec_t out = dout[idx];
        
        if constexpr (std::is_same_v<T, int8_t>) {
            // INT8: dequantize -> compute -> quantize
            float val_x = static_cast<float>(out.x) * INT8_INV_SCALE_FACTOR;
            float val_y = static_cast<float>(out.y) * INT8_INV_SCALE_FACTOR;
            float val_z = static_cast<float>(out.z) * INT8_INV_SCALE_FACTOR;
            float val_w = static_cast<float>(out.w) * INT8_INV_SCALE_FACTOR;
            
            float scale_x = static_cast<float>(s[idx].x) * INT8_INV_SCALE_FACTOR;
            float scale_y = static_cast<float>(s[idx].y) * INT8_INV_SCALE_FACTOR;
            float scale_z = static_cast<float>(s[idx].z) * INT8_INV_SCALE_FACTOR;
            float scale_w = static_cast<float>(s[idx].w) * INT8_INV_SCALE_FACTOR;
            
            dout[idx].x = static_cast<int8_t>(__float2int_rn(val_x * inv_mean * scale_x * INT8_SCALE_FACTOR));
            dout[idx].y = static_cast<int8_t>(__float2int_rn(val_y * inv_mean * scale_y * INT8_SCALE_FACTOR));
            dout[idx].z = static_cast<int8_t>(__float2int_rn(val_z * inv_mean * scale_z * INT8_SCALE_FACTOR));
            dout[idx].w = static_cast<int8_t>(__float2int_rn(val_w * inv_mean * scale_w * INT8_SCALE_FACTOR));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16: convert to float for computation, then back to half
            float val_x = __half2float(out.x);
            float val_y = __half2float(out.y);
            float scale_x = __half2float(s[idx].x);
            float scale_y = __half2float(s[idx].y);
            
            dout[idx].x = __float2half(val_x * inv_mean * scale_x);
            dout[idx].y = __float2half(val_y * inv_mean * scale_y);
        } else {
            // FP32: direct computation
            dout[idx].x = out.x * inv_mean * s[idx].x;
            dout[idx].y = out.y * inv_mean * s[idx].y;
            dout[idx].z = out.z * inv_mean * s[idx].z;
            dout[idx].w = out.w * inv_mean * s[idx].w;
        }
    }   
}

template <typename T>
void launchRMSNorm(TensorWrapper<T> * decoder_out,
                   TensorWrapper<T> * decoder_residual,
                   LayerNormWeight<T>& attn_norm_weight,
                   float eps,
                   bool is_last){
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int num_threads = hidden_units / 4;
    T *rsd = decoder_residual->data;
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    RMSNorm<T><<<grid, block>>>(decoder_out->data,
                               rsd,
                               attn_norm_weight.gamma,
                               eps,
                               num_tokens,
                               hidden_units);
}

template void launchRMSNorm<float>(TensorWrapper<float> * decoder_out,
                                   TensorWrapper<float> * decoder_residual,
                                   LayerNormWeight<float>& attn_norm_weight,
                                   float eps,
                                   bool is_last);
template void launchRMSNorm<half>(TensorWrapper<half> * decoder_out,
                                  TensorWrapper<half> * decoder_residual,
                                  LayerNormWeight<half>& attn_norm_weight,
                                  float eps,
                                  bool is_last);
template void launchRMSNorm<int8_t>(TensorWrapper<int8_t> * decoder_out,
                                     TensorWrapper<int8_t> * decoder_residual,
                                     LayerNormWeight<int8_t>& attn_norm_weight,
                                     float eps,
                                     bool is_last);
                                   
                                   