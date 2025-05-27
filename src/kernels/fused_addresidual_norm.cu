#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_addresidual_norm.h"

template<typename T>
__device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val; // 32 threads return val, but only 0th thread is sum val
}
// !!!when blocksize < 32, use blockDim.x/32 to get warp nums is wrong, we should ceil it instead
template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = (blockDim.x + 31) / 32;
    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    if(laneid == 0){
        warpsum[wid] = val;
    }
    __syncthreads();

    T sum = tid < warpnum ? warpsum[tid] : (T)0.0f;
    sum = warpReduceSum<T>(sum); //though 0th own the sum, but dont need to shfl sync
    return sum;
}
// 1.this kernel is used after self attention in every layer
// 2.I allocate threads number by assuming head size can be divided by 4 and 2

/////////////////////////////////////////////////////////////FP32///////////////////////////////////////////////////////////////
template<typename T>
__global__ void FusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    T* residual, 
                                    T* decoder_out, // [num tokens, hidden_units]
                                    /*optional*/const T* bias,  // [hidden_units]
                                    const T* scale, // [hidden_units], RMSNorm weights
                                    float eps, // RMSNorm eps
                                    int num_tokens, 
                                    int hidden_units){
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;

    Vec_t *rsd, *bia, *s;  
    Vec_t tmp;  

    Vec_t* de_out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);// note the offset should divide vec size

    T thread_accm = static_cast<T>(0); 

    rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);//note the offset should divide vec size

    if (bias != nullptr){ 
        bia = reinterpret_cast<Vec_t*>(const_cast<T*>(bias)); 
    } 

    ///////////////////////////////////////////////////////1.Add residual////////////////////////////////////////////////////////////////////////
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        if (residual != nullptr) { 
            de_out[i].x += rsd[i].x; 
            de_out[i].y += rsd[i].y;
            de_out[i].z += rsd[i].z;
            de_out[i].w += rsd[i].w;

            rsd[i].x = de_out[i].x;
            rsd[i].y = de_out[i].y;
            rsd[i].z = de_out[i].z;
            rsd[i].w = de_out[i].w;
        }
        //TODO: to update rsd by rsd + bias when bias is valid
        ///////////////////////////////////////////////////////2.Add bias////////////////////////////////////////////////////////////////////////
        if (bias != nullptr) {
            de_out[i].x += bia[i].x;
            de_out[i].y += bia[i].y;
            de_out[i].z += bia[i].z;
            de_out[i].w += bia[i].w;
        }
        
        thread_accm += de_out[i].x * de_out[i].x;
        thread_accm += de_out[i].y * de_out[i].y;
        thread_accm += de_out[i].z * de_out[i].z;
        thread_accm += de_out[i].w * de_out[i].w;
    } // addresidual

    ///////////////////////////////////////////////////////3.Implement RMSnorm:   RMSnorm(X) = scale * X / [(1/N) * sum_N(X^2) + \epsilon]^(1/2) ///////////////////////////////////////////
    T blocksum = blockReduceSum<T>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){ 
	    inv_fenmu = rsqrt(blocksum / hidden_units + eps);
        //debug info printf("inv_fenmu on GPU is %f\n", inv_fenmu);
    }
    __syncthreads();

    // rmsnorm
    if (scale != nullptr){
        s = reinterpret_cast<Vec_t*>(const_cast<T*>(scale));  
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        //s = reinterpret_cast<Vec_t*>(const_cast<T*>(scale))[i];
        de_out[i].x = s[i].x * de_out[i].x * inv_fenmu;   
        de_out[i].y = s[i].y * de_out[i].y * inv_fenmu;
        de_out[i].z = s[i].z * de_out[i].z * inv_fenmu;
        de_out[i].w = s[i].w * de_out[i].w * inv_fenmu;
    }
}

template<typename T>
void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    TensorWrapper<T>* residual, 
                                    TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<T>& norm,
                                    T* scale, //RMSNorm weights
                                    float eps) //RMSNorm eps
{
    int batch_size = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    T* bias = norm.bias;
    T* gamma = scale;
    int vec_size = Vec<T>::size;
    int num_threads = hidden_units / vec_size; // assume head size can be divided by 4 and 2

    //分配GridDim为bs的数量， 分配BlockDim为hiddensize/vec_size 的数量 这里的vec_size是指向量化的类型，float4时vec_size=4
    dim3 grid(batch_size);
    dim3 block(num_threads);
    FusedAddBiasResidualRMSNorm<T><<<grid, block>>>(residual->data, 
                                                decoder_out->data,
                                                bias,
                                                gamma,
                                                eps,
                                                batch_size,
                                                hidden_units);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(decoder_out->data);
#else
#endif
}
//////////////////////////////////这个kernel是FP16类型的融合残差加+RMSNorm///////////////////////////////////////
template<>
__global__ void FusedAddBiasResidualRMSNorm<half>( // residual.shape = [num tokens, hidden_units]
                                    half* residual, 
                                    half* decoder_out, // [num tokens, hidden_units]
                                    /*optional*/const half* bias,  // [hidden_units]
                                    const half* scale, // [hidden_units], RMSNorm weights
                                    float eps, // RMSNorm eps
                                    int num_tokens, 
                                    int hidden_units){
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;

    Vec_t *rsd, *bia, *s;  
    Vec_t tmp;  

    Vec_t* de_out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);

    float thread_accm = 0.0f; // Use float for accumulation to avoid overflow

    rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);

    if (bias != nullptr){ 
        bia = reinterpret_cast<Vec_t*>(const_cast<half*>(bias)); 
    } 

    ///////////////////////////////////////////////////////1.Add residual////////////////////////////////////////////////////////////////////////
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        if (residual != nullptr) { 
            de_out[i] = __hadd2(de_out[i], rsd[i]);
            rsd[i] = de_out[i];
        }
        
        ///////////////////////////////////////////////////////2.Add bias////////////////////////////////////////////////////////////////////////
        if (bias != nullptr) {
            de_out[i] = __hadd2(de_out[i], bia[i]);
        }
        
        // Convert to float for accumulation
        float x = __half2float(de_out[i].x);
        float y = __half2float(de_out[i].y);
        thread_accm += x * x + y * y;
    } // addresidual

    ///////////////////////////////////////////////////////3.Implement RMSnorm:   RMSnorm(X) = scale * X / [(1/N) * sum_N(X^2) + \epsilon]^(1/2) ///////////////////////////////////////////
    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){ 
	    inv_fenmu = rsqrtf(blocksum / hidden_units + eps);
    }
    __syncthreads();

    // rmsnorm
    if (scale != nullptr){
        s = reinterpret_cast<Vec_t*>(const_cast<half*>(scale));  
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        de_out[i] = __hmul2(__hmul2(s[i], de_out[i]), scalar_cast_vec<Vec_t>(__float2half(inv_fenmu)));
    }
}

//////////////////////////////////这个kernel是INT8类型的融合残差加+RMSNorm///////////////////////////////////////
template<>
__global__ void FusedAddBiasResidualRMSNorm<int8_t>( // residual.shape = [num tokens, hidden_units]
                                    int8_t* residual, 
                                    int8_t* decoder_out, // [num tokens, hidden_units]
                                    /*optional*/const int8_t* bias,  // [hidden_units]
                                    const int8_t* scale, // [hidden_units], RMSNorm weights
                                    float eps, // RMSNorm eps
                                    int num_tokens, 
                                    int hidden_units){
    int vec_size = Vec<int8_t>::size;
    using Vec_t = typename Vec<int8_t>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;

    Vec_t *rsd, *bia, *s;  
    Vec_t tmp;  

    Vec_t* de_out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);

    float thread_accm = 0.0f; // Use float for accumulation

    rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);

    if (bias != nullptr){ 
        bia = reinterpret_cast<Vec_t*>(const_cast<int8_t*>(bias)); 
    } 

    ///////////////////////////////////////////////////////1.Add residual////////////////////////////////////////////////////////////////////////
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        if (residual != nullptr) { 
            // Dequantize, add, then quantize back for residual
            float dx = (float)de_out[i].x * INT8_INV_SCALE_FACTOR;
            float dy = (float)de_out[i].y * INT8_INV_SCALE_FACTOR;
            float dz = (float)de_out[i].z * INT8_INV_SCALE_FACTOR;
            float dw = (float)de_out[i].w * INT8_INV_SCALE_FACTOR;
            
            float rx = (float)rsd[i].x * INT8_INV_SCALE_FACTOR;
            float ry = (float)rsd[i].y * INT8_INV_SCALE_FACTOR;
            float rz = (float)rsd[i].z * INT8_INV_SCALE_FACTOR;
            float rw = (float)rsd[i].w * INT8_INV_SCALE_FACTOR;
            
            dx += rx;
            dy += ry;
            dz += rz;
            dw += rw;
            
            de_out[i].x = (int8_t)__float2int_rn(dx * INT8_SCALE_FACTOR);
            de_out[i].y = (int8_t)__float2int_rn(dy * INT8_SCALE_FACTOR);
            de_out[i].z = (int8_t)__float2int_rn(dz * INT8_SCALE_FACTOR);
            de_out[i].w = (int8_t)__float2int_rn(dw * INT8_SCALE_FACTOR);
            
            rsd[i] = de_out[i];
        }
        
        ///////////////////////////////////////////////////////2.Add bias////////////////////////////////////////////////////////////////////////
        if (bias != nullptr) {
            // Dequantize, add bias, then quantize back
            float dx = (float)de_out[i].x * INT8_INV_SCALE_FACTOR;
            float dy = (float)de_out[i].y * INT8_INV_SCALE_FACTOR;
            float dz = (float)de_out[i].z * INT8_INV_SCALE_FACTOR;
            float dw = (float)de_out[i].w * INT8_INV_SCALE_FACTOR;
            
            float bx = (float)bia[i].x * INT8_INV_SCALE_FACTOR;
            float by = (float)bia[i].y * INT8_INV_SCALE_FACTOR;
            float bz = (float)bia[i].z * INT8_INV_SCALE_FACTOR;
            float bw = (float)bia[i].w * INT8_INV_SCALE_FACTOR;
            
            dx += bx;
            dy += by;
            dz += bz;
            dw += bw;
            
            de_out[i].x = (int8_t)__float2int_rn(dx * INT8_SCALE_FACTOR);
            de_out[i].y = (int8_t)__float2int_rn(dy * INT8_SCALE_FACTOR);
            de_out[i].z = (int8_t)__float2int_rn(dz * INT8_SCALE_FACTOR);
            de_out[i].w = (int8_t)__float2int_rn(dw * INT8_SCALE_FACTOR);
        }
        
        // Dequantize for RMSNorm computation
        float x = (float)de_out[i].x * INT8_INV_SCALE_FACTOR;
        float y = (float)de_out[i].y * INT8_INV_SCALE_FACTOR;
        float z = (float)de_out[i].z * INT8_INV_SCALE_FACTOR;
        float w = (float)de_out[i].w * INT8_INV_SCALE_FACTOR;
        
        thread_accm += x * x + y * y + z * z + w * w;
    } // addresidual

    ///////////////////////////////////////////////////////3.Implement RMSnorm:   RMSnorm(X) = scale * X / [(1/N) * sum_N(X^2) + \epsilon]^(1/2) ///////////////////////////////////////////
    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){ 
	    inv_fenmu = rsqrtf(blocksum / hidden_units + eps);
    }
    __syncthreads();

    // rmsnorm
    if (scale != nullptr){
        s = reinterpret_cast<Vec_t*>(const_cast<int8_t*>(scale));  
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        // Dequantize data and scale for RMSNorm
        float dx = (float)de_out[i].x * INT8_INV_SCALE_FACTOR;
        float dy = (float)de_out[i].y * INT8_INV_SCALE_FACTOR;
        float dz = (float)de_out[i].z * INT8_INV_SCALE_FACTOR;
        float dw = (float)de_out[i].w * INT8_INV_SCALE_FACTOR;
        
        float sx = (float)s[i].x * INT8_INV_SCALE_FACTOR;
        float sy = (float)s[i].y * INT8_INV_SCALE_FACTOR;
        float sz = (float)s[i].z * INT8_INV_SCALE_FACTOR;
        float sw = (float)s[i].w * INT8_INV_SCALE_FACTOR;
        
        // Apply RMSNorm and quantize back
        de_out[i].x = (int8_t)__float2int_rn((sx * dx * inv_fenmu) * INT8_SCALE_FACTOR);
        de_out[i].y = (int8_t)__float2int_rn((sy * dy * inv_fenmu) * INT8_SCALE_FACTOR);
        de_out[i].z = (int8_t)__float2int_rn((sz * dz * inv_fenmu) * INT8_SCALE_FACTOR);
        de_out[i].w = (int8_t)__float2int_rn((sw * dw * inv_fenmu) * INT8_SCALE_FACTOR);
    }
}

template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    TensorWrapper<float>* residual, 
                                    TensorWrapper<float>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<float>& norm,
                                    float* scale, //RMSNorm weights
                                    float eps);

template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    TensorWrapper<half>* residual, 
                                    TensorWrapper<half>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<half>& norm,
                                    half* scale, //RMSNorm weights
                                    float eps);

template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    TensorWrapper<int8_t>* residual, 
                                    TensorWrapper<int8_t>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<int8_t>& norm,
                                    int8_t* scale, //RMSNorm weights
                                    float eps);
