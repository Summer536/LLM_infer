#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <cstring>

#include "src/kernels/rmsnorm_kernel.h"

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// Helper function for rounding float to int (CPU version)
inline int cpu_float2int_rn(float val) {
    return static_cast<int>(roundf(val));
}

template<typename T>
void initializeData(T* data, int size, float scale = 1.0f) {
    for(int i = 0; i < size; i++) {
        if constexpr (std::is_same_v<T, int8_t>) {
            // For INT8, use quantized values
            float val = (float)(i % 10 + 1) * scale;
            data[i] = static_cast<int8_t>(cpu_float2int_rn(val * INT8_SCALE_FACTOR));
        } else if constexpr (std::is_same_v<T, half>) {
            // For FP16
            data[i] = __float2half((float)(i % 10 + 1) * scale);
        } else {
            // For FP32
            data[i] = static_cast<float>((i % 10 + 1)) * scale;
        }
    }
}

void CPUfusedresidandRMSNorm(float * h_decoder_out, float *h_scale, float eps, int hidden_units, int num_tokens){
    for(int b = 0; b < num_tokens; b++){
        float inv_fenmu = 0.0f;
        float mean = 0.0f;
        float input = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < hidden_units; i++){
            input = h_decoder_out[b * hidden_units + i];
            sum += input * input;
        }
        mean = (float)sum / hidden_units;
        inv_fenmu = rsqrtf(mean + eps);

        for(int i = 0; i < hidden_units; i++){
            h_decoder_out[b * hidden_units + i] = h_decoder_out[b * hidden_units + i] * inv_fenmu * h_scale[i];
        }
    }
}

template <typename T>
bool CheckResult(float *CPUoutput, T *GPUoutput, int output_size){
    float fp32GPUoutput = 0.0f;
    int errors = 0;
    constexpr int MAX_ERRORS_TO_PRINT = 10;
    
    // Set tolerance based on data type
    float tolerance;
    if constexpr (std::is_same_v<T, int8_t>) {
        tolerance = 0.1f; // Higher tolerance for INT8 due to quantization
    } else if constexpr (std::is_same_v<T, half>) {
        tolerance = 1e-1f; // Medium tolerance for FP16
    } else {
        tolerance = 1e-6f; // High precision for FP32
    }
    
    // 打印前几个结果作为参考
    printf("===== 前10个结果对比 (tolerance: %f) =====\n", tolerance);
    for(int i = 0; i < 10 && i < output_size; i++){
        if constexpr (std::is_same_v<T, int8_t>) {
            fp32GPUoutput = static_cast<float>(GPUoutput[i]) * INT8_INV_SCALE_FACTOR;
        } else if constexpr (std::is_same_v<T, half>) {
            fp32GPUoutput = __half2float(GPUoutput[i]);
        } else {
            fp32GPUoutput = static_cast<float>(GPUoutput[i]);
        }
        printf("位置 %d: CPU = %f, GPU = %f, 差异 = %f\n", 
               i, CPUoutput[i], fp32GPUoutput, fabs(CPUoutput[i] - fp32GPUoutput));
    }
    
    for(int i = 0; i < output_size; i++){
        if constexpr (std::is_same_v<T, int8_t>) {
            fp32GPUoutput = static_cast<float>(GPUoutput[i]) * INT8_INV_SCALE_FACTOR;
        } else if constexpr (std::is_same_v<T, half>) {
            fp32GPUoutput = __half2float(GPUoutput[i]);
        } else {
            fp32GPUoutput = static_cast<float>(GPUoutput[i]);
        }
        
        if(fabs(CPUoutput[i] - fp32GPUoutput) > tolerance){
            if (errors < MAX_ERRORS_TO_PRINT) {
                printf("错误位置 %d: CPU = %f, GPU = %f, 差异 = %f\n", 
                        i, CPUoutput[i], fp32GPUoutput, fabs(CPUoutput[i] - fp32GPUoutput));
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("总共发现 %d 个错误(超过%f的差异)\n", errors, tolerance);
        return false;
    }
    return true;
}

template<typename T>
void testRMSNorm(const std::string& type_name) {
    std::cout << "\n=== 测试 RMSNorm " << type_name << " ===\n" << std::endl;
    
    const int num_tokens = 4; 
    const int hidden_units = 16; 
    const int total_size = num_tokens * hidden_units;
    float eps = 1e-5;

    // 为CPU和GPU版本分配内存
    T *h_decoder_out = (T*)malloc(total_size * sizeof(T));
    float *h_decoder_out_float = (float*)malloc(total_size * sizeof(float)); // CPU computation in float
    T *gpu_result = (T*)malloc(total_size * sizeof(T)); // 用于存储GPU结果
    T *d_decoder_out;
    CHECK(cudaMalloc((void**)&d_decoder_out, total_size * sizeof(T)));
    
    // 初始化输入数据
    initializeData(h_decoder_out, total_size);
    
    // Convert to float for CPU computation
    for(int i = 0; i < total_size; i++){
        if constexpr (std::is_same_v<T, int8_t>) {
            h_decoder_out_float[i] = static_cast<float>(h_decoder_out[i]) * INT8_INV_SCALE_FACTOR;
        } else if constexpr (std::is_same_v<T, half>) {
            h_decoder_out_float[i] = __half2float(h_decoder_out[i]);
        } else {
            h_decoder_out_float[i] = static_cast<float>(h_decoder_out[i]);
        }
    }
    
    printf("初始化输入数据完成，前10个值：\n");
    for(int i = 0; i < 10 && i < total_size; i++){
        printf("%f ", h_decoder_out_float[i]);
    }
    printf("\n");

    // 为残差和缩放参数分配内存
    T* d_decoder_rsd;
    CHECK(cudaMalloc((void**)&d_decoder_rsd, total_size * sizeof(T)));
    T* h_scale = (T*)malloc(hidden_units * sizeof(T));
    float* h_scale_float = (float*)malloc(hidden_units * sizeof(T));
    T* d_scale;
    CHECK(cudaMalloc((void**)&d_scale, hidden_units * sizeof(T)));
    
    // 初始化缩放参数
    initializeData(h_scale, hidden_units);
    for(int i = 0; i < hidden_units; i++){
        if constexpr (std::is_same_v<T, int8_t>) {
            h_scale_float[i] = static_cast<float>(h_scale[i]) * INT8_INV_SCALE_FACTOR;
        } else if constexpr (std::is_same_v<T, half>) {
            h_scale_float[i] = __half2float(h_scale[i]);
        } else {
            h_scale_float[i] = static_cast<float>(h_scale[i]);
        }
    }
    
    printf("初始化缩放参数完成，前10个值：\n");
    for(int i = 0; i < 10 && i < hidden_units; i++){
        printf("%f ", h_scale_float[i]);
    }
    printf("\n");

    // 复制数据到GPU
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, total_size * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scale, h_scale, hidden_units * sizeof(T), cudaMemcpyHostToDevice));

    // 创建张量包装器
    DataType type = getTensorType<T>();
    TensorWrapper<T>* decoder_out_tensor = new TensorWrapper<T>(Device::GPU, type, {num_tokens, hidden_units}, d_decoder_out);
    TensorWrapper<T>* decoder_rsd = new TensorWrapper<T>(Device::GPU, type, {num_tokens, hidden_units}, d_decoder_rsd);
    LayerNormWeight<T> scale;
    scale.gamma = d_scale;
    
    // 运行GPU版本的RMSNorm
    std::cout << "开始执行GPU版本的RMSNorm..." << std::endl;
    launchRMSNorm(decoder_out_tensor, decoder_rsd, scale, eps);
    CHECK(cudaDeviceSynchronize()); // 确保GPU计算完成
    std::cout << "GPU版本的RMSNorm执行完成" << std::endl;
    
    // 复制GPU结果回CPU
    CHECK(cudaMemcpy(gpu_result, d_decoder_out, total_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    // 运行CPU版本的RMSNorm
    std::cout << "开始执行CPU版本的RMSNorm..." << std::endl;
    CPUfusedresidandRMSNorm(h_decoder_out_float, h_scale_float, eps, hidden_units, num_tokens);
    std::cout << "CPU版本的RMSNorm执行完成" << std::endl;
    
    // 检查结果并输出对比信息
    bool is_right = CheckResult<T>(h_decoder_out_float, gpu_result, total_size);
    std::cout << "RMSNorm " << type_name << " 测试 " << (is_right ? "通过" : "失败") << std::endl;
    
    // 清理内存
    free(h_decoder_out);
    free(h_decoder_out_float);
    free(gpu_result);
    free(h_scale);
    free(h_scale_float);
    
    delete decoder_out_tensor;
    delete decoder_rsd;
    
    cudaFree(d_decoder_out);
    cudaFree(d_decoder_rsd);
    cudaFree(d_scale);
}

int main(int argc, char* argv[]){
    std::string data_type = "fp32";
    if (argc > 1) {
        data_type = argv[1];
    }
    
    std::cout << "Testing RMSNorm with data type: " << data_type << std::endl;
    
    if (data_type == "fp32") {
        testRMSNorm<float>("FP32");
    } else if (data_type == "fp16") {
        testRMSNorm<half>("FP16");
    } else if (data_type == "int8") {
        testRMSNorm<int8_t>("INT8");
    } else {
        std::cerr << "Unknown data type: " << data_type << std::endl;
        std::cerr << "Usage: " << argv[0] << " [fp32|fp16|int8]" << std::endl;
        return 1;
    }
    
    return 0;
}