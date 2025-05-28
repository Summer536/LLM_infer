#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <cstring>     // strcmp

#include <cuda.h>
#include "src/kernels/topK.h"

// Helper function for rounding float to int (CPU version)
inline int cpu_float2int_rn(float val) {
    return static_cast<int>(roundf(val));
}

template<typename T>
void initializeTopKData(T* data, int size) {
    for(int i = 0; i < size; i++) {
        if constexpr (std::is_same_v<T, int8_t>) {
            // For INT8, simulate probability scores quantized
            float prob_score = static_cast<float>(i); // Ascending scores
            data[i] = static_cast<T>(cpu_float2int_rn(prob_score * INT8_SCALE_FACTOR));
        } else if constexpr (std::is_same_v<T, half>) {
            // For FP16
            data[i] = __float2half(static_cast<float>(i));
        } else {
            // For FP32
            data[i] = static_cast<T>(i);
        }
    }
}

template<typename T>
void printTopKResults(T* values, int* ids, int size, const char* name) {
    std::cout << name << " 结果:" << std::endl;
    for(int i = 0; i < size; i++) {
        if constexpr (std::is_same_v<T, int8_t>) {
            float val_f = static_cast<float>(values[i]) * INT8_INV_SCALE_FACTOR;
            printf("  TopK[%d]: ID = %d, 值 = %d (%.2f)\n", i, ids[i], static_cast<int>(values[i]), val_f);
        } else if constexpr (std::is_same_v<T, half>) {
            printf("  TopK[%d]: ID = %d, 值 = %.2f\n", i, ids[i], __half2float(values[i]));
        } else {
            printf("  TopK[%d]: ID = %d, 值 = %.2f\n", i, ids[i], static_cast<float>(values[i]));
        }
    }
}

template<typename T>
void testTopK(const std::string& type_name) {
    std::cout << "\n=== Testing TopK with " << type_name << " ===\n" << std::endl;
    
    const int batch_size = 1;
    const int vocab_size = 30000;
    const int beamwidth = 2;
    const int K = 5;
    const int BlockPerBeam = 8;
    
    const int probs_size = batch_size * vocab_size * beamwidth;
    const int topK_val_buf_size = batch_size * beamwidth * BlockPerBeam * K;
    const int topK_ids_buf_size = batch_size * beamwidth * BlockPerBeam * K;
    const int final_topK_val_buf_size = batch_size * beamwidth * K;
    
    std::cout << "配置参数:" << std::endl;
    std::cout << "  batch_size = " << batch_size << ", vocab_size = " << vocab_size << std::endl;
    std::cout << "  beamwidth = " << beamwidth << ", K = " << K << std::endl;
    std::cout << "  probs_size = " << probs_size << ", final_topK_size = " << final_topK_val_buf_size << std::endl;
    
    // Allocate memory for probs
    T* h_probs = (T*)malloc(sizeof(T) * probs_size);
    T *d_probs;
    cudaMalloc((void**)&d_probs, sizeof(T) * probs_size);
    
    // Allocate memory for intermediate topK results
    int *d_tmp_topk_ids;
    cudaMalloc((void**)&d_tmp_topk_ids, sizeof(int) * topK_ids_buf_size);
    T *d_tmp_topk_vals;
    cudaMalloc((void**)&d_tmp_topk_vals, sizeof(T) * topK_val_buf_size);

    // Allocate memory for final topK results
    int* h_final_topk_ids = (int*)malloc(sizeof(int) * final_topK_val_buf_size);
    int *d_final_topk_ids;
    cudaMalloc((void**)&d_final_topk_ids, sizeof(int) * final_topK_val_buf_size);

    T* h_final_topk_vals = (T*)malloc(sizeof(T) * final_topK_val_buf_size);
    T *d_final_topk_vals;
    cudaMalloc((void**)&d_final_topk_vals, sizeof(T) * final_topK_val_buf_size);

    // Initialize data
    initializeTopKData(h_probs, probs_size);
    
    std::cout << "初始化 " << probs_size << " 个概率值，前10个值:" << std::endl;
    for(int i = 0; i < 10; i++) {
        if constexpr (std::is_same_v<T, int8_t>) {
            float val_f = static_cast<float>(h_probs[i]) * INT8_INV_SCALE_FACTOR;
            printf("  probs[%d] = %d (%.2f)\n", i, static_cast<int>(h_probs[i]), val_f);
        } else if constexpr (std::is_same_v<T, half>) {
            printf("  probs[%d] = %.2f\n", i, __half2float(h_probs[i]));
        } else {
            printf("  probs[%d] = %.2f\n", i, static_cast<float>(h_probs[i]));
        }
    }
    
    // Copy probs to device
    cudaMemcpy(d_probs, h_probs, sizeof(T)*probs_size, cudaMemcpyHostToDevice);

    // Create tensor wrappers
    DataType type = getTensorType<T>();
    DataType type_int = getTensorType<int>();
    
    TensorWrapper<T>* probs_tensor = new TensorWrapper<T>(Device::GPU, 
                                                                type,
                                                                {batch_size * beamwidth, vocab_size}, 
                                                                d_probs);
    TensorWrapper<int> *tmp_topk_ids = new TensorWrapper<int>(Device::GPU, 
                                                                type_int,
                                                                {batch_size, beamwidth, BlockPerBeam, K}, 
                                                                d_tmp_topk_ids);
    TensorWrapper<T>* tmp_topk_vals = new TensorWrapper<T>(Device::GPU, 
                                                                type,
                                                                {batch_size, beamwidth, BlockPerBeam, K}, 
                                                                d_tmp_topk_vals);
    TensorWrapper<int> *final_topk_ids = new TensorWrapper<int>(Device::GPU, 
                                                                type_int,
                                                                {batch_size * beamwidth, K}, 
                                                                d_final_topk_ids);
    TensorWrapper<T> *final_topk_vals = new TensorWrapper<T>(Device::GPU, 
                                                                type,
                                                                {batch_size * beamwidth, K}, 
                                                                d_final_topk_vals);
    
    std::cout << "启动 " << type_name << " TopK kernel..." << std::endl;
    launchTopKforBeamSearch(probs_tensor, tmp_topk_ids, tmp_topk_vals, final_topk_ids, final_topk_vals);
    std::cout << "TopK kernel 执行完成" << std::endl;
    
    // Copy results back to host
    cudaMemcpy(h_final_topk_ids, d_final_topk_ids, sizeof(int) * final_topK_val_buf_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_topk_vals, d_final_topk_vals, sizeof(T) * final_topK_val_buf_size, cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\n" << type_name << " TopK 结果:" << std::endl;
    printTopKResults(h_final_topk_vals, h_final_topk_ids, final_topK_val_buf_size, "Final TopK");
    
    // Verify results - check if values are in descending order
    bool correct = true;
    for(int batch = 0; batch < batch_size * beamwidth; batch++) {
        for(int k = 1; k < K; k++) {
            int idx_curr = batch * K + k;
            int idx_prev = batch * K + k - 1;
            
            float val_curr, val_prev;
            if constexpr (std::is_same_v<T, int8_t>) {
                val_curr = static_cast<float>(h_final_topk_vals[idx_curr]) * INT8_INV_SCALE_FACTOR;
                val_prev = static_cast<float>(h_final_topk_vals[idx_prev]) * INT8_INV_SCALE_FACTOR;
            } else if constexpr (std::is_same_v<T, half>) {
                val_curr = __half2float(h_final_topk_vals[idx_curr]);
                val_prev = __half2float(h_final_topk_vals[idx_prev]);
            } else {
                val_curr = static_cast<float>(h_final_topk_vals[idx_curr]);
                val_prev = static_cast<float>(h_final_topk_vals[idx_prev]);
            }
            
            if (val_curr > val_prev) {
                std::cerr << "错误: TopK结果未按降序排列，batch " << batch << ", 位置 " << k << std::endl;
                correct = false;
                break;
            }
        }
    }
    
    if (correct) {
        std::cout << "✓ " << type_name << " TopK 测试通过 - 结果正确排序" << std::endl;
    } else {
        std::cout << "✗ " << type_name << " TopK 测试失败" << std::endl;
    }
    
    // Cleanup
    free(h_probs);
    free(h_final_topk_ids);
    free(h_final_topk_vals);
    cudaFree(d_probs);
    cudaFree(d_final_topk_ids);
    cudaFree(d_final_topk_vals);
    cudaFree(d_tmp_topk_ids);
    cudaFree(d_tmp_topk_vals);
    
    delete probs_tensor;
    delete tmp_topk_ids;
    delete tmp_topk_vals;
    delete final_topk_ids;
    delete final_topk_vals;
    
    std::cout << type_name << " TopK 测试完成！" << std::endl;
}

// there is no top k cpu kernel implementation now
// we compare the kernel correctnesss by eyes and result print infos
// Usage: `./test_topk [fp32|fp16|int8]` to test different precisions
int main(int argc, char* argv[]) {
    std::string data_type = "fp32";
    if (argc > 1) {
        data_type = argv[1];
    }
    
    std::cout << "Testing TopK with data type: " << data_type << std::endl;
    
    if (data_type == "fp32") {
        testTopK<float>("FP32");
    } else if (data_type == "fp16") {
        testTopK<half>("FP16");
    } else if (data_type == "int8") {
        testTopK<int8_t>("INT8");
    } else {
        std::cerr << "Unknown data type: " << data_type << std::endl;
        std::cerr << "Usage: " << argv[0] << " [fp32|fp16|int8]" << std::endl;
        return 1;
    }
    
    return 0;
}
