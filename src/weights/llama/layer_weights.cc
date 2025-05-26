#include <random>
#include "src/weights/llama/layer_weights.h"
#include "src/utils/macro.h"

template<typename T>
LlamaLayerWeight<T>::LlamaLayerWeight(int     head_num,  
                                    int     kv_head_num,
                                    int     head_size,
                                    int     inter_size,
                                    WeightType weight_type,
                                    bool       attn_bias):
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    hidden_units(head_num * head_size),
    inter_size(inter_size),
    weight_type(weight_type),
    attn_bias(attn_bias)
{   
    // 1.attn_norm_weight只需要对(/src/weights/llama/norm_weights.h) T* gamma;给分配好，初始化一下即可！
    CHECK(cudaMalloc((void**)&attn_norm_weight.gamma, sizeof(T) * hidden_units)); //shape is [hidden_units]

    // 2.FFN里面的RMSNorm
    CHECK(cudaMalloc((void**)&ffn_norm_weight.gamma, sizeof(T) * hidden_units));  //shape is [hidden_units]
    
    // 3.self_attn_weight是一个base weight，需要对(/src/weights/llama/base_weights.h) std::vector<int> shape;   T* data;    WeightType type;    T* bias;四个量进行初始化！ 
    // 3.1 qkv的weights 以下上三行分别是对WeightType type、 std::vector<int> shape 、  T* data三个进行初始化
    self_attn_weight.qkv.type = weight_type;
    self_attn_weight.qkv.shape = {(head_num + 2 * kv_head_num) * head_size, hidden_units};
    CHECK(cudaMalloc((void**)&self_attn_weight.qkv.data, sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size)); //shape is [(head_num + 2 * kv_head_num) * head_size, hidden_units]
    // 3.2 output的weights
    self_attn_weight.output.type = weight_type;
    self_attn_weight.output.shape = {hidden_units, hidden_units};
    CHECK(cudaMalloc((void**)&self_attn_weight.output.data, sizeof(T) * hidden_units * hidden_units));
    // 如果有bias，需要对bias进行分配
    if (attn_bias) {
        CHECK(cudaMalloc((void**)&self_attn_weight.qkv.bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size));  //这个bias的shape是weights的最后一个维度
        CHECK(cudaMalloc((void**)&self_attn_weight.output.bias, sizeof(T) * hidden_units)); //这个bias的shape是weights的最后一个维度
    }

    // 4.ffn_weight是一个base weight，需要对(/src/weights/llama/base_weights.h) std::vector<int> shape;   T* data;    WeightType type;    T* bias;四个量进行初始化！
    //   这里的gate和up是合在一起的
    ffn_weight.gateAndup.type = weight_type;
    ffn_weight.down.type = weight_type;
    ffn_weight.gateAndup.shape = {2 * inter_size, hidden_units};
    // ffn_weight.up.shape = {hidden_units, inter_size};
    ffn_weight.down.shape = {hidden_units, inter_size};
    CHECK(cudaMalloc((void**)&ffn_weight.gateAndup.data, sizeof(T) * hidden_units * 2 * inter_size));
    // CHECK(cudaMalloc((void**)&ffn_weight.up.data, hidden_units * inter_size));
    CHECK(cudaMalloc((void**)&ffn_weight.down.data, sizeof(T) * hidden_units * inter_size));
}

////////////////////////////////////////////////////给这些weights的数据进行初始化，并且把它拷到GPU上面去//////////////////////////////////////////////////////////////////////////
///////////////////////////////////因为这些weights是从huggingface上面下载下来的，需要将这些weights先读取进来(cpu)上，然后再把它拷到GPU上面去参与我们实际的计算。//////////////////////////

//////////////////////////下面定义了两个loadWeights的函数，一个是从huggingface上面下载下来的weights，一个是从本地加载的weights。//////////////////////////

/////////////这个函数是load从huggingface上面下载下来的weights//////////////////////////
template<typename T>
void LlamaLayerWeight<T>::loadWeights(std::string weight_path, WeightType weight_type) 
{   
    //loadWeightFromBin是一个struct结构(/src/utils/weight_utils.cu)，它里面有一个静态函数internalFunc：它接收的参数为：1.在构造函数分配好的buffer的指针 2.一系列的weights的shape 3.它们的路径
    //<T, float>：我们需要用的类型是T,python转好的类型是float的，因此这里做一个转换
    loadWeightFromBin<T, float>::internalFunc(attn_norm_weight.gamma, {hidden_units}, weight_path + ".input_layernorm.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(ffn_norm_weight.gamma, {hidden_units}, weight_path + ".post_attention_layernorm.weight.bin");

    loadWeightFromBin<T, float>::internalFunc(self_attn_weight.qkv.data, {(head_num + 2 * kv_head_num) * head_size, hidden_units}, weight_path + ".self_attn.qkv.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(self_attn_weight.output.data, {hidden_units, hidden_units}, weight_path + ".self_attn.o_proj.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(ffn_weight.gateAndup.data, {2 * inter_size, hidden_units}, weight_path + ".mlp.gate_up_proj.weight.bin");
    // loadWeightFromBin<T, float>::internalFunc(ffn_weight.up.data, {hidden_units, inter_size}, weight_path + ".mlp.up_proj.weight.bin");
    loadWeightFromBin<T, float>::internalFunc(ffn_weight.down.data, {hidden_units, inter_size}, weight_path + ".mlp.down_proj.weight.bin");
    if (attn_bias) {//TODO
        loadWeightFromBin<T, float>::internalFunc(self_attn_weight.qkv.bias, {(head_num + 2 * kv_head_num) * head_size}, weight_path + ".attention.wqkv.bias.bin");
        loadWeightFromBin<T, float>::internalFunc(self_attn_weight.output.bias, {head_num *  head_size}, weight_path + ".attention.wo.bias.bin");
    } else { //如果没有bias 则设置为空指针。这个bias都是需要根据特别模型的特别写的！像llama2就没有bias
    	self_attn_weight.qkv.bias = nullptr;
	self_attn_weight.output.bias = nullptr;
	ffn_weight.down.bias = nullptr;
    } 
}

/////////////这个函数是load本地加载的weights(仅用于测试代码正确与否以及测试性能，不关注精度)//////////////////////////
template<typename T>
void LlamaLayerWeight<T>::loadWeights() 
{   //创建一些虚假的数据weight以供测试
    T* d_dummy_attn_norm_weight;
    T* d_dummy_ffn_norm_weight;
    T* d_dummy_qkv_weights;
    //T* d_dummy_qkv_bias;
    T* d_dummy_output_weights;
    T* d_dummy_output_bias;
    T* d_dummy_ffn_down;
    T* d_dummy_ffn_down_bias;
    T* d_dummy_ffn_gate_up;
    // T* d_dummy_ffn_up;
    CHECK(cudaMalloc((void**)&d_dummy_attn_norm_weight, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_norm_weight, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_qkv_weights, sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size));
   // CHECK(cudaMalloc((void**)&d_dummy_qkv_bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size));
    CHECK(cudaMalloc((void**)&d_dummy_output_weights, sizeof(T) * hidden_units * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_output_bias, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_down, sizeof(T) * hidden_units * inter_size));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_down_bias, sizeof(T) * hidden_units));
    CHECK(cudaMalloc((void**)&d_dummy_ffn_gate_up, sizeof(T) * hidden_units * 2 * inter_size));
    // CHECK(cudaMalloc(&d_dummy_ffn_up, sizeof(T) * hidden_units * inter_size));

    T* h_dummy_attn_norm_weight = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_ffn_norm_weight = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_qkv_weights = (T*)malloc(sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size);
   // T* h_dummy_qkv_bias = (T*)malloc(sizeof(T) * (head_num + 2 * kv_head_num) * head_size);
    T* h_dummy_output_weights = (T*)malloc(sizeof(T) * hidden_units * hidden_units);
    T* h_dummy_output_bias = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_ffn_down = (T*)malloc(sizeof(T) * hidden_units * inter_size);
    T* h_dummy_ffn_down_bias = (T*)malloc(sizeof(T) * hidden_units);
    T* h_dummy_ffn_gate_up = (T*)malloc(sizeof(T) * hidden_units * 2 * inter_size);
    // T* h_dummy_ffn_up = (T*)malloc(sizeof(T) * hidden_units * inter_size);

    //初始化cpu上的值
    for (int i = 0; i < hidden_units; i++){
        h_dummy_attn_norm_weight[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_ffn_norm_weight[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_output_bias[i] = (T)(rand() % 100 / (float)100000);
        h_dummy_ffn_down_bias[i] = (T)(rand() % 100 / (float)100000);
    }
    //for (int i = 0; i < (head_num + 2 * kv_head_num) * head_size; i++) {
    //    h_dummy_qkv_bias[i] = (T)(rand() % 100 / (float)100000);
    //}
    for (int i = 0; i < hidden_units * inter_size; i++) {
        h_dummy_ffn_down[i] = (T)(rand() % 100 / (float)100000);
    }
    for (int i = 0; i < hidden_units * 2 * inter_size; i++) {   
        h_dummy_ffn_gate_up[i] = (T)(rand() % 100 / (float)100000);
        // h_dummy_ffn_up[i] = (T)1.0f;
    }
    for (int i = 0; i < hidden_units * hidden_units; i++) {
        h_dummy_output_weights[i] = (T)(rand() % 100 / (float)100000);
    }
    for (int i = 0; i < hidden_units * (head_num + 2 * kv_head_num) * head_size; i++) {
        h_dummy_qkv_weights[i] = (T)(rand() % 100 / (float)100000);
    }

    CHECK(cudaMemcpy(d_dummy_attn_norm_weight, h_dummy_attn_norm_weight, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_norm_weight, h_dummy_ffn_norm_weight, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_qkv_weights, h_dummy_qkv_weights, sizeof(T) * hidden_units * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice));
    //CHECK(cudaMemcpy(d_dummy_qkv_bias, h_dummy_qkv_bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_weights, h_dummy_output_weights, sizeof(T) * hidden_units * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_output_bias, h_dummy_output_bias, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down, h_dummy_ffn_down, sizeof(T) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_down_bias, h_dummy_ffn_down_bias, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dummy_ffn_gate_up, h_dummy_ffn_gate_up, sizeof(T) * hidden_units * 2 * inter_size, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_dummy_ffn_up, h_dummy_ffn_up, sizeof(T) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    attn_norm_weight.gamma = d_dummy_attn_norm_weight;
    ffn_norm_weight.gamma = d_dummy_ffn_norm_weight;
    self_attn_weight.qkv.data = d_dummy_qkv_weights;
    self_attn_weight.qkv.bias = nullptr;
    self_attn_weight.output.data = d_dummy_output_weights;
    self_attn_weight.output.bias = d_dummy_output_bias;
    ffn_weight.gateAndup.data = d_dummy_ffn_gate_up;
    //ffn_weight.up.data = d_dummy_ffn_up;
    ffn_weight.down.data = d_dummy_ffn_down;
    ffn_weight.down.bias = d_dummy_ffn_down_bias;
}

////////////////////////////////////////////////////weights的buffer free//////////////////////////////////////////////////////////////////////////
template<typename T>
void freeWeights(BaseWeight<T>& weights) 
{
    cudaFree(weights.data);
    if(weights.bias != nullptr) {
        cudaFree(weights.bias);
    }

    weights.data = nullptr;
    weights.bias = nullptr;
}
template<typename T>
LlamaLayerWeight<T>::~LlamaLayerWeight() 
{
    // free norm weights ptr
    cudaFree(attn_norm_weight.gamma);
    cudaFree(ffn_norm_weight.gamma);
    // free other weights, including data and bias
    freeWeights(self_attn_weight.qkv);
    freeWeights(self_attn_weight.output);
    freeWeights(ffn_weight.gateAndup);
    // freeWeights(ffn_weight.up);
    freeWeights(ffn_weight.down);
}
// template instantial required in linking time
template class LlamaLayerWeight<float>;
template class LlamaLayerWeight<half>;
