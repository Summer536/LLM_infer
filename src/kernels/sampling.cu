#include <iostream>
#include "src/kernels/sampling.h"
// mini-softmax + curand_sample
// input: [bs, K] from topK output
// output: [bs]

template<typename T>
__global__ void SamplingKernel(int* topk_id,
                               T* topk_val, //[bs, K] from topK
                               int* output_id, //[bs]
                               int* seqlen, //cumulated seq len,[bs]
                               bool* is_finished, //[bs]
                               int K,
                               int rand_num, // step
                               int end_id, // when initialize llama model, we will init it, and this is a fixed val
                               int vocab_size)
{   
    int batch_id = blockIdx.x;
    int bid = batch_id;
    int tid = threadIdx.x;
    int offset = batch_id * K + tid; 

    // Convert to float for computation, especially important for INT8
    float max_val_f;
    if constexpr (std::is_same_v<T, int8_t>) {
        max_val_f = static_cast<float>(topk_val[batch_id * K]) * INT8_INV_SCALE_FACTOR;
    } else if constexpr (std::is_same_v<T, half>) {
        max_val_f = __half2float(topk_val[batch_id * K]);
    } else {
        max_val_f = static_cast<float>(topk_val[batch_id * K]);
    }
    
    // Apply softmax: subtract max and take exp, store back in appropriate format
    float current_val_f;
    if constexpr (std::is_same_v<T, int8_t>) {
        current_val_f = static_cast<float>(topk_val[offset]) * INT8_INV_SCALE_FACTOR;
    } else if constexpr (std::is_same_v<T, half>) {
        current_val_f = __half2float(topk_val[offset]);
    } else {
        current_val_f = static_cast<float>(topk_val[offset]);
    }
    
    float exp_val = expf(current_val_f - max_val_f);
    
    // Store back in original format
    if constexpr (std::is_same_v<T, int8_t>) {
        topk_val[offset] = static_cast<T>(__float2int_rn(exp_val * INT8_SCALE_FACTOR));
    } else if constexpr (std::is_same_v<T, half>) {
        topk_val[offset] = __float2half(exp_val);
    } else {
        topk_val[offset] = static_cast<T>(exp_val);
    }
    
    __shared__ float thredhold, sum;
    if(tid == 0) {
        sum = 0.0f;
        for(int i = 0; i < K; i++) {
            float val_f;
            if constexpr (std::is_same_v<T, int8_t>) {
                val_f = static_cast<float>(topk_val[batch_id * K + i]) * INT8_INV_SCALE_FACTOR;
            } else if constexpr (std::is_same_v<T, half>) {
                val_f = __half2float(topk_val[batch_id * K + i]);
            } else {
                val_f = static_cast<float>(topk_val[batch_id * K + i]);
            }
            sum += val_f;
        }

        curandState_t state;  
        curand_init((unsigned long long)rand_num,(unsigned long long)bid, (unsigned long long)0, &state);

        thredhold = (float)curand_uniform(&state) * sum; 

        output_id[bid] = topk_id[bid * K] % vocab_size; 

        for(int i = 0; i < K; i++) {
            float val_f;
            if constexpr (std::is_same_v<T, int8_t>) {
                val_f = static_cast<float>(topk_val[batch_id * K + i]) * INT8_INV_SCALE_FACTOR;
            } else if constexpr (std::is_same_v<T, half>) {
                val_f = __half2float(topk_val[batch_id * K + i]);
            } else {
                val_f = static_cast<float>(topk_val[batch_id * K + i]);
            }
            
            thredhold = thredhold - val_f;
            if(thredhold < 0) { 
                output_id[bid] = topk_id[batch_id * K + i] % vocab_size; 
                break;
            }
        }
        seqlen[bid] = is_finished[bid] ? seqlen[bid] : seqlen[bid] + 1; 
        is_finished[bid] = output_id[bid] == end_id ? 1 : 0; 
    }
}

template<typename T>
void launchSampling(TensorWrapper<int>* topk_id, 
                    TensorWrapper<T>* topk_val, 
                    TensorWrapper<int>* seqlen, 
                    TensorWrapper<bool>* is_finished, 
                    TensorWrapper<int>* output_id,   
                    IntDict& params) { 
    int batch_size = topk_id->shape[0];
    int K = topk_id->shape[1];
    int vocab_size = params["vocab_size"]; 
    int step = params["step"];
    int end_id = params["end_id"];
    
    dim3 grid(batch_size);
    dim3 block(K); 
    SamplingKernel<<<grid, block>>>(
        topk_id->data,
        topk_val->data,
        output_id->data,
        seqlen->data,
        is_finished->data,
        K,
        step,
        end_id,
        vocab_size
    );
}

template void launchSampling(TensorWrapper<int>* topk_id,
                            TensorWrapper<float>* topk_val,
                            TensorWrapper<int>* seqlen,
                            TensorWrapper<bool>* is_finished,
                            TensorWrapper<int>* output_id,
                            IntDict& params);

template void launchSampling(TensorWrapper<int>* topk_id,
                            TensorWrapper<half>* topk_val,
                            TensorWrapper<int>* seqlen,
                            TensorWrapper<bool>* is_finished,
                            TensorWrapper<int>* output_id,
                            IntDict& params);

template void launchSampling(TensorWrapper<int>* topk_id,
                            TensorWrapper<int8_t>* topk_val,
                            TensorWrapper<int>* seqlen,
                            TensorWrapper<bool>* is_finished,
                            TensorWrapper<int>* output_id,
                            IntDict& params);
