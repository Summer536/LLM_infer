#include <math.h>
#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/qkv_bias_and_RoPE.h"

inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
{
    float2 rot_v; 
    rot_v.x = coef.x * data - coef.y * data_rotate; // cos() * x_0 - sin() * x_64
    rot_v.y = coef.x * data_rotate + coef.y * data; // cos() * x_64 + sin() * x_0
    return rot_v;
}

template <typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T *q_buf,
                                                   T *k_buf,
                                                   T *v_buf,
                                                   T *QKV,
                                                   /*optional*/const T *qkv_bias,
                                                   const int *padding_offset, // created before qkv linear
                                                   const int *history_length,
                                                   const int *input_length, // actual length of each seq
                                                   const int batch_size,
                                                   const int seq_len, // max_seq_len to pad to
                                                   const int token_num,
                                                   const int head_num,
                                                   const int kv_head_num,
                                                   const int head_size,
                                                   const int rotary_embedding_dim,
                                                   float rotary_embedding_base, // default 10000 in llama
                                                   int max_position_embeddings, /*default 2048 in llama*/
                                                   bool use_dynamic_ntk /*placeholder for ntk RoPE*/)
{
    int token_id = blockIdx.x;
    int head_id = blockIdx.y; 
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];  //需要读取offset，该量由之前的kernel计算而得(/src/kernels/cal_padding_offset_kernel.cu)


    // 1. prapare rebuilding , do rebuild padding and transpose when store
    int dst_token_id = token_id + token_padding_offset; 
    int batch_id = dst_token_id / seq_len;      
    int local_token_id = dst_token_id % seq_len; 


    // 2. bias add (llama2不需要加bias)
    int qkv_head_num = head_num + 2 * kv_head_num; 
    int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid;                                                  //当前线程访问的q id
    int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size;                           //当前线程访问的k id
    int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size + kv_head_num * head_size; //当前线程访问的v id
                                                                                
    float v = QKV[v_id]; 
    int dst_q_id = batch_id * seq_len * head_num * head_size +
                   head_id * seq_len * head_size +
                   local_token_id * head_size + tid;

    int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
                    head_id * seq_len * head_size +
                    local_token_id * head_size + tid;
    if (head_id < kv_head_num) 
    { 
        v_buf[dst_kv_id] = v;
    }


    // 3. RoPE
    const int cur_seq_history_len = history_length[batch_id];                  //当前句子历史的长度：表示当前序列之前已经处理过的历史对话长度；在多轮对话中，这个值会累积增加
    const int context_length = cur_seq_history_len + input_length[batch_id];   //当前句子历史的长度 + 当前句子的长度 = 当前句子上下文的长度
    const int timestep = cur_seq_history_len + local_token_id;                 //当前句子历史的长度 + 现在的token id = 这个token在整个对话历史中的全局位置

    if (tid >= rotary_embedding_dim / 2)
    {
        return;
    } 

    float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, timestep); 
    float2 q_rotate = GetRoPEres(QKV[q_id], QKV[q_id + head_size / 2], cos_sin);    
    float2 k_rotate = GetRoPEres(QKV[k_id], QKV[k_id + head_size / 2], cos_sin);


    // 4. write back 
    q_buf[dst_q_id] = q_rotate.x;                 
    q_buf[dst_q_id + head_size / 2] = q_rotate.y;  
    if (head_id < kv_head_num) 
    { 
        k_buf[dst_kv_id] = k_rotate.x;
        k_buf[dst_kv_id + head_size / 2] = k_rotate.y;
    }
}

// input: qkv_buf : qkv continouns buf when no padding  
// output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
//         k/v shape = [bs, kv head num, seqlen, head size]
template <typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T> *q_buf,   
                                           TensorWrapper<T> *k_buf,
                                           TensorWrapper<T> *v_buf,
                                           TensorWrapper<T> *QKV,   
                                           BaseWeight<T> &qkv,       
                                           TensorWrapper<int> *padding_offset,  
                                           TensorWrapper<int> *history_length,
                                           TensorWrapper<int> *input_length,
                                           LLaMAAttentionStaticParams &params)
{
    int token_num = QKV->shape[0];    // QKV shape = [num_tokens, qkv_head_num, head_size]
    int qkv_head_num = QKV->shape[1];
    int head_size = QKV->shape[2];
    int batch_size = q_buf->shape[0]; // q shape = [bs, head num, seqlen, head size]
    int head_num = q_buf->shape[1];
    int seq_len = q_buf->shape[2];
    int kv_head_num = (qkv_head_num - head_num) / 2;

    dim3 grid(token_num, head_num);    
    dim3 block(head_size); 
    add_fusedQKV_bias_transpose_kernel<T><<<grid, block>>>(q_buf->data,  // output
                                                           k_buf->data,  // output
                                                           v_buf->data,  // output
                                                           QKV->data,    // input
                                                           /*optional*/qkv.bias,    
                                                           padding_offset->data,    
                                                           history_length->data,   
                                                           input_length->data,
                                                           batch_size,
                                                           seq_len,
                                                           token_num,
                                                           head_num,
                                                           kv_head_num,
                                                           head_size,
                                                           params.rotary_embedding_dim,
                                                           params.rotary_embedding_base,
                                                           params.max_position_embeddings,
                                                           params.use_dynamic_ntk);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(q_buf->data);
#else
#endif
}

template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<float> *q_buf,
                                                    TensorWrapper<float> *k_buf,
                                                    TensorWrapper<float> *v_buf,
                                                    TensorWrapper<float> *QKV,
                                                    BaseWeight<float> &qkv,
                                                    TensorWrapper<int> *padding_offset,
                                                    TensorWrapper<int> *history_length,
                                                    TensorWrapper<int> *input_length,
                                                    LLaMAAttentionStaticParams &params);
template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<half> *q_buf,
                                                    TensorWrapper<half> *k_buf,
                                                    TensorWrapper<half> *v_buf,
                                                    TensorWrapper<half> *QKV,
                                                    BaseWeight<half> &qkv,
                                                    TensorWrapper<int> *padding_offset,
                                                    TensorWrapper<int> *history_length,
                                                    TensorWrapper<int> *input_length,
                                                    LLaMAAttentionStaticParams &params);



//////////////////////////////////////////////////////////////这个是对于self-decoder的，与上方的context有些区别，因此重写kernel/////////////////////////////////////////////////////////////
///不同点：1. - Context Decoder: 处理打包的QKV大矩阵   - Self Decoder: 直接处理分离的Q和K
///       2. - Context Decoder: timestep = cur_seq_history_len + local_token_id   - Self Decoder: step - 1 （当前步骤减1）
///       3.- Context Decoder: 通过判断 head_id < kv_head_num 来处理     - Self Decoder: 通过计算 kv_head_id = q_head_id / (head_num / kv_head_num)
///       4.- Context Decoder: 需要处理padding和转置     - Self Decoder: 更简单的内存访问模式
//////////////////////////////- 不需要处理padding，因为是逐token生成  - 不需要添加bias（好像上面也没添加，可能是llama的特性或者忘了），因为输入已经是处理过bias的Q和K////////////////////


template<typename T>
__global__ void rope_kernel_for_self_decoder(T* q,
                    T* k,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){
    int tid = threadIdx.x;
    int q_head_id = blockIdx.x;
    int q_batch_id = blockIdx.y;
    int kv_head_id = q_head_id / (head_num / kv_head_num);  
    int kv_batch_id = q_batch_id;

    int batch_stride = head_num * head_size;   
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;

    if (tid >= rotary_embedding_dim / 2) {
        return;
    }

    // 3.RoPE
    float k_reg = k[k_offset];
    float k_rotate_reg = k[k_offset + head_size / 2];
    float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, step - 1); //注意这里是step-1(与huggingface对比)
    float2 q_rotate = GetRoPEres(q[q_offset], q[q_offset + head_size / 2], cos_sin);
    float2 k_rotate = make_float2(0,0);
    k_rotate.x = cos_sin.x * k_reg - cos_sin.y * k_rotate_reg;
    k_rotate.y = cos_sin.x * k_rotate_reg + cos_sin.y * k_reg;

    // 4.write back
    q[q_offset] = q_rotate.x;
    q[q_offset + head_size / 2] = q_rotate.y;
    k[k_offset] = k_rotate.x;
    k[k_offset + head_size / 2] = k_rotate.y;
}

template<typename T>
void launchRoPE(TensorWrapper<T>* qkv_buf,
                TensorWrapper<int>* step,   
                LLaMAAttentionStaticParams& static_params){
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    int head_num = 32; // only for llama   
    const int head_size = qkv_buf->shape[2];
    LLM_CHECK(batch_size == 1);
    LLM_CHECK(qkv_head_num == 96);     
    LLM_CHECK(head_size == 128);
    const int cur_step = step->getVal();    

    T* qkv_data = qkv_buf->data;
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;

    int   rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int   max_position_embeddings = static_params.max_position_embeddings;

    dim3 grid(head_num, batch_size); 
    dim3 block(head_size); 
    rope_kernel_for_self_decoder<T><<<grid, block>>>(q,
                                                    k,
                                                    batch_size,
                                                    head_num,
                                                    head_num, // only for llama, kv head = head
                                                    head_size,
                                                    cur_step,
                                                    rotary_embedding_dim,
                                                    rotary_embedding_base);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(q);
#else
#endif
}

template void launchRoPE(TensorWrapper<float>* qkv_buf,
                        TensorWrapper<int>* step,
                        LLaMAAttentionStaticParams& static_params);
template void launchRoPE(TensorWrapper<half>* qkv_buf,
                        TensorWrapper<int>* step,
                        LLaMAAttentionStaticParams& static_params);

//////////////////////////////////////////////////////////////这个是对于self-decoder的，与上方的context有些区别，因此重写kernel/////////////////////////////////////////////////////////////