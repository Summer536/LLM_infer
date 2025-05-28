#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <cstring>     // strcmp

#include <math.h>
#include "src/kernels/repeat_kv.h"

// Helper function to initialize data based on type
template<typename T>
void initializeData(T* data, int size, float scale = 1.0f) {
    for(int i = 0; i < size; i++) {
        if constexpr (std::is_same_v<T, int8_t>) {
            // For INT8, use small integer values
            data[i] = static_cast<int8_t>((i % 256) - 128);
        } else if constexpr (std::is_same_v<T, half>) {
            // For FP16
            data[i] = __float2half(static_cast<float>(i) * scale);
        } else {
            // For FP32
            data[i] = static_cast<float>(i) * scale;
        }
    }
}

template<typename T>
void printResults(T* data, int size, const char* name) {
    std::cout << name << " results:" << std::endl;
    for(int i = 0; i < std::min(size, 16); i++) {
        if constexpr (std::is_same_v<T, int8_t>) {
            printf("%s[%d] = %d\n", name, i, static_cast<int>(data[i]));
        } else if constexpr (std::is_same_v<T, half>) {
            printf("%s[%d] = %f\n", name, i, __half2float(data[i]));
        } else {
            printf("%s[%d] = %f\n", name, i, data[i]);
        }
    }
}

template<typename T>
void testRepeatKVCache() {
    const int batch_size = 1;
    const int head_num = 2;
    const int kv_head_num = 2;
    const int max_seq_len = 4;
    const int max_k_len = 2;
    const int head_size = 2;
    const int num_layers = 2;
    const int k_size = num_layers * batch_size * kv_head_num * max_seq_len * head_size;
    const int out_k_size = batch_size * head_num * max_k_len * head_size;
    
    T* h_k;
    T* d_k;
    h_k = (T*)malloc(sizeof(T) * k_size);
    cudaMalloc((void**)&d_k, sizeof(T) * k_size);
    T* h_v;
    T* d_v;
    h_v = (T*)malloc(sizeof(T) * k_size);
    cudaMalloc((void**)&d_v, sizeof(T) * k_size);
    int* h_ctx_len;
    int* d_ctx_len;
    h_ctx_len = (int*)malloc(sizeof(int) * batch_size);
    cudaMalloc((void**)&d_ctx_len, sizeof(int) * batch_size);
    T* h_trans_k;
    T* d_trans_k;
    h_trans_k = (T*)malloc(sizeof(T) * out_k_size);
    cudaMalloc((void**)&d_trans_k, sizeof(T) * out_k_size);
    T* h_trans_v;
    T* d_trans_v;
    h_trans_v = (T*)malloc(sizeof(T) * out_k_size);
    cudaMalloc((void**)&d_trans_v, sizeof(T) * out_k_size);   

    // Initialize data
    initializeData(h_k, k_size);
    initializeData(h_v, k_size);
    
    int* h_layer_id = (int*)malloc(sizeof(int)*batch_size);

    for(int i = 0; i < batch_size; i++) {
       h_ctx_len[i] = 2;
       h_layer_id[i] = 0;
    }    
    
    cudaMemcpy(d_k, h_k, sizeof(T) * k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(T) * k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx_len, h_ctx_len, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    
    DataType type = getTensorType<T>(); 
    DataType type_int = getTensorType<int>(); 
    TensorWrapper<T>* in_k = new TensorWrapper<T>(Device::GPU, type, {num_layers, batch_size, kv_head_num, max_seq_len, head_size}, d_k);
    TensorWrapper<T>* in_v = new TensorWrapper<T>(Device::GPU, type, {num_layers, batch_size, kv_head_num, max_seq_len, head_size}, d_v);
    TensorWrapper<int>* ctx_len = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_ctx_len);
    TensorWrapper<T>* out_k = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size}, d_trans_k);
    TensorWrapper<T>* out_v = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size}, d_trans_v);
    TensorWrapper<int>* layer_id = new TensorWrapper<int>(Device::CPU, type_int, {batch_size}, h_layer_id);
    
    std::cout << "before launch repeat kv kernel" << std::endl;
    launchRepeatKVCache(in_k, in_v, ctx_len, layer_id, out_k, out_v);
    std::cout << "after launch repeat kv kernel" << std::endl;
    
    // Copy results back and print
    cudaMemcpy(h_trans_k, out_k->data, sizeof(T) * out_k_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_trans_v, out_v->data, sizeof(T) * out_k_size, cudaMemcpyDeviceToHost);
    
    printResults(h_trans_k, out_k_size, "k_trans");
    printResults(h_trans_v, out_k_size, "v_trans");
    
    // Cleanup
    free(h_k);
    free(h_v);
    free(h_ctx_len);
    free(h_trans_k);
    free(h_trans_v);
    free(h_layer_id);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_ctx_len);
    cudaFree(d_trans_k);
    cudaFree(d_trans_v);
    
    delete in_k;
    delete in_v;
    delete ctx_len;
    delete out_k;
    delete out_v;
    delete layer_id;
}

// there is no repeat kv cpu kernel implementation now
// we compare the kernel correctnesss by eyes
// Usage: `./test_repeat_kv [fp32|fp16|int8]` to test different precisions
int main(int argc, char* argv[]) {
    std::string data_type = "fp32";
    if (argc > 1) {
        data_type = argv[1];
    }
    
    std::cout << "Testing RepeatKVCache with data type: " << data_type << std::endl;
    
    if (data_type == "fp32") {
        testRepeatKVCache<float>();
    } else if (data_type == "fp16") {
        testRepeatKVCache<half>();
    } else if (data_type == "int8") {
        testRepeatKVCache<int8_t>();
    } else {
        std::cerr << "Unknown data type: " << data_type << std::endl;
        std::cerr << "Usage: " << argv[0] << " [fp32|fp16|int8]" << std::endl;
        return 1;
    }
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
