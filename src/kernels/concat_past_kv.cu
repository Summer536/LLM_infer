// rope之后，k/v shape = [bs, kv_head num, max_q_len, head size] 
// kv cache shape = [num layers, bs, kv_head num, max_seq_len, head size] = >[bs, kv_head num, seqlen[history_len: history_len + max q len] , head size]

#include "src/kernels/concat_past_kv.h"
#include "src/utils/cuda_debug_utils.cuh"
#include <iostream>
template <typename T> 
__global__ void append_key_cache(T *k_dst, // [num layers, bs, kv head num, max_q_len, head size]
                                 const size_t layer_offset,
                                 const T *k_src, // [bs, kv_head num, max_q_len, head size]
                                 const int kv_head_num,
                                 const int head_size,
                                 const int *cur_query_length, 
                                 const int *history_length,   
                                 const int max_q_len,
                                 const int max_seq_len)
{   
    int batch_id = blockIdx.y;   //bs
    int head_id = blockIdx.z;    //head num
    int tid = threadIdx.x;
    int token_id = blockIdx.x;   //sqn_len个Token

    // Now we are at layer_offset, which is the start of the current layer's k cache
    T *k_cache_dst = k_dst + layer_offset;

    int cur_seq_len = cur_query_length[batch_id];
    int cumsum_seq_len = history_length[batch_id]; 
    if (token_id < cur_seq_len) 
    {
        int src_offset = batch_id * kv_head_num * max_q_len * head_size + 
                         head_id * max_q_len * head_size +
                         token_id * head_size + tid;

        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size +
                         head_id * max_seq_len * head_size +
                         (cumsum_seq_len + token_id) * head_size + tid; //写到history_len开始的地方
        k_cache_dst[dst_offset] = k_src[src_offset]; //将K矩阵中的值写入到KV cache中，拼进去
    }
}

template <typename T> 
__global__ void append_value_cache(T *v_dst,
                                   const size_t layer_offset,
                                   const T *v_src,
                                   const int kv_head_num,
                                   const int head_size,
                                   const int *cur_query_length,
                                   const int *history_length,
                                   const int max_q_len,
                                   const int max_seq_len)
{
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int tid = threadIdx.x;
    int token_id = blockIdx.x;

    // Now we are at layer_offset, which is the start of the current layer's v cache
    T *v_cache_dst = v_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id];
    int cumsum_seq_len = history_length[batch_id];
    if (token_id < cur_seq_len)
    {
        int src_offset = batch_id * kv_head_num * max_q_len * head_size +
                         head_id * max_q_len * head_size +
                         token_id * head_size + tid;
        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size +
                         head_id * max_seq_len * head_size +
                         (cumsum_seq_len + token_id) * head_size + tid;
        v_cache_dst[dst_offset] = v_src[src_offset];
    }
}

template <typename T>
void launchConcatKVCache(//输入
                         TensorWrapper<T> *k_src, // from qkv bias and rope {batch_size, kv_head_num, max_q_len, head_size}
                         TensorWrapper<T> *v_src,
                         TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                                                            //查询当前的layerid 是layer0 还是layer1 ......
                         TensorWrapper<int> *cur_query_length, // current epoch or local input length,[batchsize]
                                                            //当前输入的句子长度
                         TensorWrapper<int> *history_length,//保存着每个句子的历史上下文长度
                         //输出
                         TensorWrapper<T> *k_dst, //{num_layers, batch_size, kv_head_num, max_seq_len, head_size}
                         TensorWrapper<T> *v_dst) //输出k\v cache
{   
    int batch_size = k_src->shape[0]; 
    int max_seq_len = k_dst->shape[3];
    int kv_head_num = k_src->shape[1];
    int max_q_len = k_src->shape[2];
    int head_size = k_src->shape[3];
    int blockSize = head_size;

    int layer = layer_id->getVal(); 
    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size; 

    dim3 grid(max_q_len, batch_size, kv_head_num);
    append_key_cache<T><<<grid, blockSize>>>(k_dst->data,
                                             layer_offset,
                                             k_src->data,
                                             kv_head_num,
                                             head_size,
                                             /*(int*)*/ cur_query_length->data,
                                             /*(int*)*/ history_length->data,
                                             max_q_len,
                                             max_seq_len);

    append_value_cache<T><<<grid, blockSize>>>(v_dst->data,
                                               layer_offset,
                                               v_src->data,
                                               kv_head_num,
                                               head_size,
                                               /*(int*)*/ cur_query_length->data,
                                               /*(int*)*/ history_length->data,
                                               max_q_len,
                                               max_seq_len);

}

template void launchConcatKVCache(TensorWrapper<float> *k_src, // from qkv bias and rope
                                  TensorWrapper<float> *v_src,
                                  TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                                  TensorWrapper<int> *cur_query_length, // current epoch or local input length,[batchsize]
                                  TensorWrapper<int> *history_length,
                                  TensorWrapper<float> *k_dst,
                                  TensorWrapper<float> *v_dst);

template void launchConcatKVCache(TensorWrapper<half> *k_src, // from qkv bias and rope
                                  TensorWrapper<half> *v_src,
                                  TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                                  TensorWrapper<int> *cur_query_length, // current epoch or local input length,[batchsize]
                                  TensorWrapper<int> *history_length,
                                  TensorWrapper<half> *k_dst,
                                  TensorWrapper<half> *v_dst);

template void launchConcatKVCache(TensorWrapper<int8_t> *k_src, // from qkv bias and rope
                                  TensorWrapper<int8_t> *v_src,
                                  TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                                  TensorWrapper<int> *cur_query_length, // current epoch or local input length,[batchsize]
                                  TensorWrapper<int> *history_length,
                                  TensorWrapper<int8_t> *k_dst,
                                  TensorWrapper<int8_t> *v_dst);
