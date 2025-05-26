#include <math.h>
#include "src/utils/debug_utils.h"
#include "src/layers/attention/masked_self_attention.h"

template<typename T>
LLaMASelfAttentionLayer<T>::LLaMASelfAttentionLayer( 
                               int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator):
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator),
    hidden_units(head_num * head_size),
    attn_static_params(attn_params),
    q_head_per_kv(head_num / kv_head_num),
    scale(float(1 / sqrt(head_size))){}

template<typename T>
void LLaMASelfAttentionLayer<T>::allocForForward(LLaMAAttentionDynParams& params) {  //LLaMAAttentionDynParams:(位于/src/models/llama/llama_params.h)
    int batch_size = params.batch_size;
    int num_tokens = params.num_tokens;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>(); 
    const int qkv_head_num = head_num + 2 * kv_head_num;
    //当前step的q k v的shape里面step或seqlen都是1，之前step的kv在做gemv的时候直接从kv cache拿

    qkv_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, qkv_head_num, head_size});  //为什么是qkv headnum？因为我们算的是qkv gemm，一共是三个gemm的大矩阵，所以我们要准备三个buffer并将其拼接为一个大buffer
    mha_output = new TensorWrapper<T>(Device::GPU, type, {batch_size, hidden_units});
    
    qkv_buf->data = allocator->Malloc(qkv_buf->data, sizeof(T) * batch_size * qkv_head_num * head_size, false);
    mha_output->data = allocator->Malloc(
        mha_output->data, sizeof(T) * batch_size * hidden_units, false);
}

template<typename T>
void LLaMASelfAttentionLayer<T>::freeBuf(){
    allocator->Free(qkv_buf->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(mha_output->data);
    DeviceSyncAndCheckCudaError();
}

// params order of launcher function in LaMAContextAttentionLayer<T>::forward: (input[Tensor], input[Tensor],...,weight[Weight], output[*])
template<typename T>
void LLaMASelfAttentionLayer<T>::forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params)
{  
    allocForForward(params); 

    //1. qkv linear
    //shape:[bs,1,q_hidden_units] * [q_hidden_units, hidden_units] = [bs,1,hidden_units]
    Tensor* attention_input = inputs["attention_input"];
    launchLinearGemm(attention_input->as<T>(), weights.qkv, qkv_buf, cublas_wrapper, false, true);

    DeviceSyncAndCheckCudaError();

    //2. biasRope   //这里Rope和下面的融合算子分开写也是因为llama2的Rope的特殊的地方
    Tensor* attention_output = outputs["attention_output"];
    // kv cache shape = [bs, kv head num, max seq len head size]
    Tensor* key_cache       = outputs["all_k_cache"];  //kv cache是要从outputs里面获取的，因为它是作为输出的，因为我们每次产生一个token之后，都要产生一个k和v供下次attention来用
    Tensor* value_cache     = outputs["all_v_cache"];  
    Tensor* finished = inputs["finished"]; 
    Tensor* step = inputs["step"];
    Tensor* layer_id = inputs["layer_id"];

    launchRoPE(qkv_buf, step->as<int>(), attn_static_params);

    DeviceSyncAndCheckCudaError();

    // 3. fused decoder self attention
    launchDecoderMaskedMHA<T>(qkv_buf, weights.qkv, layer_id->as<int>(), key_cache->as<T>(), value_cache->as<T>(), finished->as<bool>(), step->as<int>(), mha_output, attn_static_params);

    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    save_tensor(mha_output ,"self_decoder_qk_v_after_bmm.bin", layer_id->as<int>());
#else
#endif

    // 4. attention output linear
    launchLinearGemm(mha_output, weights.output, attention_output->as<T>(), cublas_wrapper, false, true);

    DeviceSyncAndCheckCudaError();

#ifdef SAVE_DATA
    save_tensor(mha_output ,"self_decoder_outlinear_out.bin", layer_id->as<int>());
#else
#endif
    this->freeBuf();
}

template class LLaMASelfAttentionLayer<float>;
template class LLaMASelfAttentionLayer<half>;
