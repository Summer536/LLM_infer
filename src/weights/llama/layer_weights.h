#pragma once
#include "src/weights/llama/norm_weights.h"
#include "src/weights/llama/attention_weights.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/utils/weight_utils.h"
template<typename T>
class LlamaLayerWeight {
private:
    int     head_num;
    int     kv_head_num;
    int     head_size;
    int     hidden_units;
    int     inter_size;
    WeightType weight_type;
    int     bit_size;
    bool    attn_bias; //为了区分qkv gemm后面有没有bias，默认为false，即没有bias

public:
    LlamaLayerWeight() = delete;
    LlamaLayerWeight(int head_num,  
                    int  kv_head_num,
                    int  head_size,
                    int  inter_size,
                    WeightType weight_type,
                    bool attn_bias);
    ~LlamaLayerWeight();

    void loadWeights(std::string weight_path, WeightType weight_type);
    
    void loadWeights(); 

    LayerNormWeight<T> attn_norm_weight;      //attention之前的RMSnorm的gamma               (/src/weights/llama/norm_weights.h)
    LayerNormWeight<T> ffn_norm_weight;       //gate uplinear之前的RMSnorm的gamma           (/src/weights/llama/norm_weights.h)
    LLaMAattentionWeights<T> self_attn_weight;//qkv gemm 和outputlinear的weight和bias       (/src/weights/llama/attention_weights.h)
    LLaMAFFNWeights<T> ffn_weight;            //gate up Down linear的weight和bias           (/src/weights/llama/ffn_weights.h)
    //为什么embedding的weights以及LMhead(linear)的weights没有放在这里表示呢？
    //因为这两个不太属于layerweights，因为上述的四种weights都是由transformer的N层layer堆叠起来的
    //    而embedding和LMhead(linear)它始终只有一种，不管你有几种layer。
};
