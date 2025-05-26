#include <iostream>
#include "src/utils/macro.h"
#include "src/utils/debug_utils.h"
#include "src/layers/decoder/context_decoder.h"

/////////////////////////////////////////////////////////////alloc buffer///////////////////////////////////////////////////////////////////////
template<typename T>
void LlamaContextDecoder<T>::allocForForward(LLaMAAttentionDynParams& params)
{   
    int num_tokens = params.num_tokens;
    int batch_size = params.batch_size;
    int max_q_len = params.max_q_len; 
    int max_k_len = params.max_k_len; 

    DataType type = getTensorType<T>(); 
    DataType type_int = getTensorType<int>(); 
    decoder_residual = new TensorWrapper<T>(Device::GPU, type, {num_tokens, hidden_units});
    attention_mask = new TensorWrapper<T>(Device::GPU, type, {batch_size, max_q_len, max_k_len});
    padding_offset = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, max_q_len});
    cum_seqlens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size + 1});
    decoder_residual->data = allocator->Malloc(decoder_residual->data, sizeof(T) * num_tokens * hidden_units, false);
    attention_mask->data = allocator->Malloc(attention_mask->data, sizeof(T) * batch_size * max_q_len * max_k_len, false);
    padding_offset->data = allocator->Malloc(padding_offset->data, sizeof(int) * batch_size * max_q_len, false);
    cum_seqlens->data = allocator->Malloc(cum_seqlens->data, sizeof(int) * (batch_size + 1), false);
}

/////////////////////////////////////////////////////////////free buffer///////////////////////////////////////////////////////////////////////
template<typename T>
void LlamaContextDecoder<T>::freeBuf()
{
    allocator->Free(attention_mask->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(padding_offset->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(cum_seqlens->data);
    DeviceSyncAndCheckCudaError();
}

///////////////////////////////////////////////////////////// froward ///////////////////////////////////////////////////////////////////////
template<typename T>
void LlamaContextDecoder<T>::forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight<T>*>& layerWeights, TensorMap& output_tensors, LLaMAAttentionDynParams& dyn_params)
{                                   
    allocForForward(dyn_params); 

    Tensor* seq_lens = input_tensors["input_length"]; 
    // 1. calculate padding offset
    launchCalPaddingoffset(padding_offset, //out
                           cum_seqlens, //out
                           seq_lens->as<int>()); // in

    DeviceSyncAndCheckCudaError();

    //2. build causal mask
    Tensor* context_length = input_tensors["context_length"];
    launchBuildCausalMasks<T>(attention_mask, //out
                            seq_lens->as<int>(), //in //q, input lens, [bs]
                            context_length->as<int>());//in //k, context lens, [bs]

    DeviceSyncAndCheckCudaError();

    // 3. context attn
    Tensor* history_length = input_tensors["history_length"];
    Tensor* decoder_output = output_tensors["decoder_output"];
    Tensor* all_k_cache = output_tensors["all_k_cache"];
    Tensor* all_v_cache = output_tensors["all_v_cache"];
    DataType type_int = getTensorType<int>();
    DataType type = getTensorType<T>();
    Tensor* layer_id = input_tensors["layer_id"];
    Tensor* decoder_input = input_tensors["decoder_input"];
    LLM_CHECK_WITH_INFO(decoder_input->as<T>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!"); //任取两个量检查一下它们的指针别为空
    LLM_CHECK_WITH_INFO(history_length->as<int>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    
    ///由于要在这个函数中调用下级的context attention layer和ffn layer的forward函数，因此要使用上面取出来的量构建两个新的TensorMap。
    TensorMap ctx_attn_inputs{
        {"attention_input", decoder_input},
        {"padding_offset", padding_offset},
        {"history_length", history_length},
        {"input_length", seq_lens},
        {"context_length", context_length},
        {"attention_mask", attention_mask},
        {"layer_id", layer_id}
    };
    TensorMap ctx_attn_outputs{
        {"attention_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    /////////////////32层的context decoder layer （包括RMSnorm + context attention layer + FusedaddbiasResidualAndRMSnorm + FFN layer + AddbiasandResidual）
    // 32layer都是用的同一个layer的buffer，直接malloc一层所需的buffer即可，32层会进行复用。省显存！
    
    for(int layer_id = 0; layer_id < num_layer; layer_id++) {//num_layer; layer_id++) {
        
        if (layer_id > 0){
            TensorWrapper<int>* layer = new TensorWrapper<int>(Device::CPU, type_int, {1}, &layer_id);
            ctx_attn_inputs.insert("layer_id", layer); 
        }

        //这里的decoder_input如果时for的第一次循环，那么是正常读取外面传进来的值，如果是第二次即以上，传进来的这个input其实是上一次for循环的输出（它在这个for循环的最后一行做了更新）
        //因此这样就能串起来32层的context decoder layer
        decoder_input = ctx_attn_inputs["attention_input"];

        ///3.1 RMSnorm
        launchRMSNorm(decoder_input->as<T>(), //in&out, [num tokens, q_hidden_units]
                    decoder_residual, // = rmsnorm input hidden states, used for next add residual
                    layerWeights[layer_id]->attn_norm_weight,//rmsnorm weights, [q_hidden_units]  //(位于/src/weights/llama/layer_weigths.h)中定义好的RMSnorm的weights
                    rmsnorm_eps);

        DeviceSyncAndCheckCudaError();  

        ///3.2 context attention layer (/src/layers/attention/context_attention.h)
        ctxAttn->forward(ctx_attn_inputs, ctx_attn_outputs, layerWeights[layer_id]->self_attn_weight, dyn_params, ctxAttn->GetAttnStaticParams()); 
        
        ///3.3 FusedaddbiasResidualAndRMSnorm
        launchFusedAddBiasResidualRMSNorm(decoder_residual, //in residual from tensor before rmsnorm and return decoder_residual + decoder_output, [num tokens, hidden_units]
                                        decoder_output->as<T>(), //in&out from attention output, [num tokens, hidden_units]
                                        layerWeights[layer_id]->self_attn_weight.output, //bias 
                                        layerWeights[layer_id]->ffn_norm_weight.gamma,//rmsnorm weights, [hidden_units] 
                                        rmsnorm_eps);

        DeviceSyncAndCheckCudaError();

        #ifdef SAVE_DATA
            save_tensor(decoder_output->as<T>() ,"ffn_input.bin", layer_id);
        #else
        #endif

        ///3.4 FFN layer (/src/layers/ffn/ffn.h)
        TensorMap ffn_inputs{
            {"ffn_input", decoder_output}
        };
        TensorMap ffn_outputs{
            {"ffn_output", decoder_output} 
        };
	    dyn_params.is_ctx = true; 
        ffn->forward(ffn_inputs, ffn_outputs, layerWeights[layer_id]->ffn_weight, dyn_params);

        #ifdef SAVE_DATA
            save_tensor(decoder_output->as<T>() ,"ffn_output.bin", layer_id);
        #else
        #endif 

        ///3.5 AddbiasandResidual
        launchAddResidual(decoder_residual, //residual, [num tokens, hidden_units]
                        decoder_output->as<T>() //in&out, [num tokens, hidden_units]
                        );
        DeviceSyncAndCheckCudaError();

        ////////////把当前layer的输出作为下一次layer的输入！/////////////////////
        ctx_attn_inputs.insert("attention_input", decoder_output);
    }


    freeBuf();
    DeviceSyncAndCheckCudaError();
}

template class LlamaContextDecoder<float>;
template class LlamaContextDecoder<half>;
