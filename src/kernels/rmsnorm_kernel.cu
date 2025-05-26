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

        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y;
        thread_sum += vec.z * vec.z;
        thread_sum += vec.w * vec.w;
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
        dout[idx].x = out.x * inv_mean * s[idx].x;
        dout[idx].y = out.y * inv_mean * s[idx].y;
        dout[idx].z = out.z * inv_mean * s[idx].z;
        dout[idx].w = out.w * inv_mean * s[idx].w;
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
                                   
                                   