#pragma once
#include <string>
#include "src/weights/weight.h"
#include "src/weights/base_weights.h"
#include "src/weights/llama/embedding_weights.h"
#include "src/weights/llama/layer_weights.h"

template<typename T>
struct LlamaWeight : public Weight {
private:   
    int     hidden_units;
    int     inter_size;
    int     vocab_size;
    int     vocab_size_padded;
    int     num_layer;
    WeightType weight_type;
    
public:   
    std::vector<LlamaLayerWeight<T>*> llama_layer_weight; //llama_layer_weight(Lesson23节做好的)  32层的weight  
    LayerNormWeight<T> out_rmsnorm_weight;                //context decoder中的最后一个RMSnorm
    EmbeddingWeight<T> post_decoder_embedding_weight;     //Sampling里面的LLMhead linear
    EmbeddingWeight<T> pre_decoder_embedding_weight;      //context decoder中第一个embedding weights
    
    LlamaWeight() = default;
    LlamaWeight( 
        int     head_num,
        int     kv_head_num,
        int     head_size,
        int     inter_size,
        int     vocab_size,
        int     num_layer,
        bool    attn_bias,
        WeightType weight_type       
    );
    ~LlamaWeight(); //把拥有的buffer free掉，避免内存泄漏
    void loadWeights(std::string weight_path); //加载一些真实的weight
    void loadWeightsFromDummy(); //加载一些虚假的weight，用于测试程序正确性以及速度
};