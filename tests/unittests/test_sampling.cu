#include <iostream>
#include <cstring>
#include "src/kernels/sampling.h"
#include "src/utils/macro.h"

// Helper function for rounding float to int (CPU version)
inline int cpu_float2int_rn(float val) {
    return static_cast<int>(roundf(val));
}

template<typename T>
void initializeSamplingData(T* data, int size) {
    for(int i = 0; i < size; i++) {
        if constexpr (std::is_same_v<T, int8_t>) {
            // For INT8, simulate log probabilities quantized
            float log_prob = static_cast<float>(5 - (i % 5)); // Values like 5, 4, 3, 2, 1
            data[i] = static_cast<T>(cpu_float2int_rn(log_prob * INT8_SCALE_FACTOR));
        } else if constexpr (std::is_same_v<T, half>) {
            // For FP16, log probabilities
            data[i] = __float2half(static_cast<float>(5 - (i % 5)));
        } else {
            // For FP32, log probabilities
            data[i] = static_cast<T>(5 - (i % 5));
        }
    }
}

template<typename T>
void testSampling(const std::string& type_name) {
    std::cout << "\n=== Testing Sampling with " << type_name << " ===\n" << std::endl;
    
    const int batch_size = 3;
    const int K = 3;
    int vocab_size = 1000;
    int step = 6;
    int end_id = 10;

    // Allocate memory
    int *h_topkid = (int *)malloc(sizeof(int) * batch_size * K);
    int *d_topkid;
    cudaMalloc((void **)&d_topkid, sizeof(int) * batch_size * K);
    
    T *h_topkval = (T *)malloc(sizeof(T) * batch_size * K);
    T *d_topkval;
    cudaMalloc((void **)&d_topkval, sizeof(T) * batch_size * K);
    
    int *h_outid = (int *)malloc(sizeof(int) * batch_size);
    int *d_outid;
    cudaMalloc((void **)&d_outid, sizeof(int) * batch_size);
    
    int *h_cuseqlen = (int *)malloc(sizeof(int) * batch_size);
    int *d_cuseqlen;
    cudaMalloc((void **)&d_cuseqlen, sizeof(int) * batch_size);
    
    bool *h_finished = (bool *)malloc(sizeof(bool) * batch_size);
    bool *d_finished;
    cudaMalloc((void **)&d_finished, sizeof(bool) * batch_size);
    
    // Initialize data
    for (int i = 0; i < batch_size; i++) {
        h_finished[i] = false;
        h_cuseqlen[i] = 4;
    }
    
    for (int i = 0; i < batch_size * K; i++) {
        h_topkid[i] = i;
    }
    
    // Initialize values based on type
    initializeSamplingData(h_topkval, batch_size * K);
    
    printf("初始化TopK值：\n");
    for(int i = 0; i < batch_size * K; i++) {
        if constexpr (std::is_same_v<T, int8_t>) {
            printf("topk_val[%d] = %d (%.2f)\n", i, static_cast<int>(h_topkval[i]), 
                   static_cast<float>(h_topkval[i]) * INT8_INV_SCALE_FACTOR);
        } else if constexpr (std::is_same_v<T, half>) {
            printf("topk_val[%d] = %.2f\n", i, __half2float(h_topkval[i]));
        } else {
            printf("topk_val[%d] = %.2f\n", i, static_cast<float>(h_topkval[i]));
        }
    }
    
    // Copy to device
    CHECK(cudaMemcpy(d_topkval, h_topkval, sizeof(T) * batch_size * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_topkid, h_topkid, sizeof(int) * batch_size * K, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_cuseqlen, h_cuseqlen, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_finished, h_finished, sizeof(bool) * batch_size, cudaMemcpyHostToDevice));
    
    // Create tensor wrappers
    DataType type = getTensorType<T>();
    DataType type_int = getTensorType<int>();
    DataType type_bool = getTensorType<bool>();
    
    TensorWrapper<T> *topk_val = new TensorWrapper<T>(Device::GPU, type, {batch_size, K}, d_topkval);
    TensorWrapper<int> *topk_id = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, K}, d_topkid);
    TensorWrapper<int> *cuseqlen = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_cuseqlen);
    TensorWrapper<bool> *finished = new TensorWrapper<bool>(Device::GPU, type_bool, {batch_size}, d_finished);
    TensorWrapper<int> *output_id = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_outid);
    
    IntDict intParams;
    intParams.insert({"step", step});
    intParams.insert({"vocab_size", vocab_size});
    intParams.insert({"end_id", end_id});
    
    std::cout << "启动 " << type_name << " sampling kernel..." << std::endl;
    launchSampling<T>(topk_id, topk_val, cuseqlen, finished, output_id, intParams);
    std::cout << "Sampling kernel 执行完成" << std::endl;
    
    // Copy results back
    CHECK(cudaMemcpy(h_outid, output_id->data, sizeof(int) * batch_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_cuseqlen, cuseqlen->data, sizeof(int) * batch_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_finished, finished->data, sizeof(bool) * batch_size, cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << "\n结果：" << std::endl;
    for (int i = 0; i < batch_size; i++) {
        std::cout << "序列 " << i + 1 << ": 输出ID = " << h_outid[i] 
                  << ", 序列长度 = " << h_cuseqlen[i] 
                  << ", 是否结束 = " << (h_finished[i] ? "是" : "否") << std::endl;
    }
    
    // Cleanup
    free(h_topkid);
    free(h_topkval);
    free(h_finished);
    free(h_cuseqlen);
    free(h_outid);
    cudaFree(d_topkid);
    cudaFree(d_topkval);
    cudaFree(d_finished);
    cudaFree(d_cuseqlen);
    cudaFree(d_outid);
    
    delete topk_val;
    delete topk_id;
    delete cuseqlen;
    delete finished;
    delete output_id;
    
    std::cout << type_name << " sampling 测试完成！" << std::endl;
}

// there is no CPU implementation of this kernel
// we compare the kernel correctnesss by eyes and result print infos
// Usage: `./test_sampling [fp32|fp16|int8]` to test different precisions
int main(int argc, char *argv[]) {
    std::string data_type = "fp32";
    if (argc > 1) {
        data_type = argv[1];
    }
    
    std::cout << "Testing Sampling with data type: " << data_type << std::endl;
    
    if (data_type == "fp32") {
        testSampling<float>("FP32");
    } else if (data_type == "fp16") {
        testSampling<half>("FP16");
    } else if (data_type == "int8") {
        testSampling<int8_t>("INT8");
    } else {
        std::cerr << "Unknown data type: " << data_type << std::endl;
        std::cerr << "Usage: " << argv[0] << " [fp32|fp16|int8]" << std::endl;
        return 1;
    }
    
    return 0;
}
