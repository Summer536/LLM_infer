#include "src/kernels/repeat_kv.h"
#include "src/utils/cuda_debug_utils.cuh"
#include <iostream>
// if MQA or GQA, we should use this repeat kv kernel to broadcast kv head num to q head num

template <typename T>
__global__ void repeat_value_cache(T *v_dst, 
                                   const T *v_src, 
                                   const size_t layer_offset,
                                   const int head_num,
                                   const int q_head_per_kv, //每个kv_head_num对应多少个q_head_num
                                   const int head_size,
                                   const int *context_length, 
                                   const int max_k_len,
                                   const int max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const auto val_src = v_src + layer_offset;   //KV cache shape [num_layers, bs, kv_head_num, max_seq_len, head_size]
    const auto val_dst = v_dst;                  //output shape [bs, q_head_num,  max_seq_len, head_size]

    const auto seq_len = context_length[batch_id];

    const int v_head_size_id = idx % head_size; 
    const int v_seq_len_id = idx / head_size;  

    // only fetch context_length(<max_seq_len) kv data from all kv cache of current seq
    if (v_seq_len_id < seq_len) 
    {   
        const int64_t src_idx = batch_id * (head_num / q_head_per_kv) * head_size * max_seq_len + 
                                head_id / q_head_per_kv * head_size * max_seq_len +                
                                v_seq_len_id * head_size +                                         
                                v_head_size_id;                                                   
        const int64_t dst_idx = batch_id * head_num * head_size * max_k_len +    
                                head_id * head_size * max_k_len +            
                                v_seq_len_id * head_size +                   
                                v_head_size_id;                              
        val_dst[dst_idx] = val_src[src_idx];
    }
}
template <typename T>
void launchRepeatKVCache(//输入
                         TensorWrapper<T> *k_cache_src, //{num_layers, batch_size, kv_head_num, max_seq_len, head_size}
                         TensorWrapper<T> *v_cache_src, //{num_layers, batch_size, kv_head_num, max_seq_len, head_size}
                         TensorWrapper<int> *context_length, //文本的长度
                         TensorWrapper<int> *layer_id, //查询当前的layerid 是layer0 还是layer1 ......

                         //输出
                         TensorWrapper<T> *k_cache_dst, //输出广播后得kv矩阵//{batch_size, head_num, max_k_len, head_size}
                         TensorWrapper<T> *v_cache_dst)
{
    int batch_size = context_length->shape[0];
    int kv_head_num = k_cache_src->shape[2]; 
    int max_seq_len = k_cache_src->shape[3];
    int head_num = k_cache_dst->shape[1];

    int max_k_len = k_cache_dst->shape[2];
    int head_size = k_cache_dst->shape[3];

    int layer = layer_id->getVal();

    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size; 
    int q_head_per_kv = head_num / kv_head_num; 

    int blockSize = 128;   
    dim3 block(blockSize);
    dim3 grid((max_k_len * head_size + blockSize - 1) / blockSize, batch_size, head_num);
    repeat_value_cache<T><<<grid, block>>>(v_cache_dst->data,
                                           v_cache_src->data,
                                           layer_offset,
                                           head_num,
                                           q_head_per_kv,
                                           head_size,
                                           context_length->data,
                                           max_k_len,
                                           max_seq_len);

    repeat_value_cache<T><<<grid, block>>>(k_cache_dst->data,
                                           k_cache_src->data,
                                           layer_offset,
                                           head_num,
                                           q_head_per_kv,
                                           head_size,
                                           context_length->data,
                                           max_k_len,
                                           max_seq_len);

#ifdef PRINT_DATA
    print_data<<<1, 1>>>(k_cache_dst->data);
#else
#endif
}

template void launchRepeatKVCache(TensorWrapper<float> *k_cache_src,
                                  TensorWrapper<float> *v_cache_src,
                                  TensorWrapper<int> *context_length,
                                  TensorWrapper<int> *layer_id,
                                  TensorWrapper<float> *k_cache_dst,
                                  TensorWrapper<float> *v_cache_dst);
template void launchRepeatKVCache(TensorWrapper<half> *k_cache_src,
                                  TensorWrapper<half> *v_cache_src,
                                  TensorWrapper<int> *context_length,
                                  TensorWrapper<int> *layer_id,
                                  TensorWrapper<half> *k_cache_dst,
                                  TensorWrapper<half> *v_cache_dst);
template void launchRepeatKVCache(TensorWrapper<int8_t> *k_cache_src,
                                  TensorWrapper<int8_t> *v_cache_src,
                                  TensorWrapper<int> *context_length,
                                  TensorWrapper<int> *layer_id,
                                  TensorWrapper<int8_t> *k_cache_dst,
                                  TensorWrapper<int8_t> *v_cache_dst);
