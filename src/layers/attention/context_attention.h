#pragma once
#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h"
#include "src/kernels/attn_softmax_kernel.h"
#include "src/kernels/qkv_bias_and_RoPE.h"
#include "src/kernels/fused_transpose_and_remv_pad.h"
#include "src/kernels/concat_past_kv.h"
#include "src/kernels/repeat_kv.h"
#include "src/utils/tensor.h"
#include "src/kernels/cublas_utils.h"
#include "src/models/llama/llama_params.h"

///////////////////////////////////////////整体实现思路是将layer封装成了一个类class/////////////////////////////////////////////
template<typename T>
class LLaMAContextAttentionLayer {
private:
    // this params are shared across all LLMs
    const int head_num;  
    const int head_size;
    const int hidden_units; //hidden_units = head_num * head_size

    const int q_head_per_kv; // q_head_per_kv = q_head_num / kv_head_num
    const int kv_head_num;

    // scale = sqrt(head_size)
    float scale; 

    // this params are only saw in llama and are unchanged 
    LLaMAAttentionStaticParams attn_static_params;  

    cudaStream_t stream;

    // allocator的指针，用于下方的allocForForward函数对下方的几个TensorWrapper<T>*分配内存
    BaseAllocator* allocator; //位于(/sec/memory/allocator/base_allocator.h)中，allocator是指向BaseAllocator对象的指针

    cublasWrapper* cublas_wrapper;

    TensorWrapper<T>*  qkv_buf_wo_pad = nullptr;      
    TensorWrapper<T>*  q_buf_w_pad = nullptr;
    TensorWrapper<T>*  k_buf_w_pad = nullptr;
    TensorWrapper<T>*  v_buf_w_pad = nullptr;
    TensorWrapper<T>*  k_cache_buf = nullptr;
    TensorWrapper<T>*  v_cache_buf = nullptr;
    TensorWrapper<T>*  qk_buf = nullptr;
    TensorWrapper<T>*  qkv_buf_w_pad = nullptr;
    TensorWrapper<T>*  qkv_buf_wo_pad_1 = nullptr;      

public:
    ///////////////////////////////一些成员函数(public)///////////////////////////////////
    // 1.构造函数
    LLaMAContextAttentionLayer(int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator);
    LLaMAAttentionStaticParams& GetAttnStaticParams(){
        return attn_static_params;
    }
    // 2.分配forward所需那些PPT上的九个中间变量的buffer
    void allocForForward(LLaMAAttentionDynParams& params);
    // 3.forward结束后释放所占用中间变量的buffer，除了输入输出buffer
    void freeBuf();
    // 4.forward，即推理过程函数
    void forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params, LLaMAAttentionStaticParams& static_params);
};
