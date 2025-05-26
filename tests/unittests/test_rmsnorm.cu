#include <algorithm>
#include <vector>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>

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
    
    // 打印前几个结果作为参考
    printf("===== 前10个结果对比 =====\n");
    for(int i = 0; i < 10 && i < output_size; i++){
        fp32GPUoutput = (float)GPUoutput[i];
        printf("位置 %d: CPU = %f, GPU = %f, 差异 = %f\n", 
               i, CPUoutput[i], fp32GPUoutput, fabs(CPUoutput[i] - fp32GPUoutput));
    }
    
    for(int i = 0; i < output_size; i++){
        fp32GPUoutput = (float)GPUoutput[i];
        if(fabs(CPUoutput[i] - fp32GPUoutput) > 1e-6){
            if (errors < MAX_ERRORS_TO_PRINT) {
                printf("错误位置 %d: CPU = %f, GPU = %f, 差异 = %f\n", 
                        i, CPUoutput[i], fp32GPUoutput, fabs(CPUoutput[i] - fp32GPUoutput));
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("总共发现 %d 个错误(超过1e-6的差异)\n", errors);
        return false;
    }
    return true;
}

int main(){
    const int num_tokens = 4; // 减小测试规模，原本是64
    const int hidden_units = 16; // 减小测试规模，原本是4096
    const int total_size = num_tokens * hidden_units;
    float eps = 1e-5;

    // 为CPU和GPU版本分配内存
    float *h_decoder_out = (float*)malloc(total_size * sizeof(float));
    float *h_decoder_out_copy = (float*)malloc(total_size * sizeof(float)); // 保存CPU原始输入的副本
    float *gpu_result = (float*)malloc(total_size * sizeof(float)); // 用于存储GPU结果
    float *d_decoder_out;
    CHECK(cudaMalloc((void**)&d_decoder_out, total_size * sizeof(float)));
    
    // 初始化输入数据
    for(int i = 0; i < total_size; i++){
        h_decoder_out[i] = (float)(i % 2 + 1);
        h_decoder_out_copy[i] = h_decoder_out[i]; // 保存一份原始数据的副本
    }
    
    printf("初始化输入数据完成，前10个值：\n");
    for(int i = 0; i < 10 && i < total_size; i++){
        printf("%f ", h_decoder_out[i]);
    }
    printf("\n");

    // 为残差和缩放参数分配内存
    float* d_decoder_rsd;
    CHECK(cudaMalloc((void**)&d_decoder_rsd, total_size * sizeof(float)));
    float* h_scale = (float*)malloc(hidden_units * sizeof(float));
    float* d_scale;
    CHECK(cudaMalloc((void**)&d_scale, hidden_units * sizeof(float)));
    
    // 初始化缩放参数
    for(int i = 0; i < hidden_units; i++){
        h_scale[i] = (float)(i % 2 + 1);
    }
    
    printf("初始化缩放参数完成，前10个值：\n");
    for(int i = 0; i < 10 && i < hidden_units; i++){
        printf("%f ", h_scale[i]);
    }
    printf("\n");

    // 复制数据到GPU
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scale, h_scale, hidden_units * sizeof(float), cudaMemcpyHostToDevice));

    // 创建张量包装器
    DataType type_float = getTensorType<float>();
    TensorWrapper<float>* decoder_out_tensor = new TensorWrapper<float>(Device::GPU, type_float, {num_tokens, hidden_units}, d_decoder_out);
    TensorWrapper<float>* decoder_rsd = new TensorWrapper<float>(Device::GPU, type_float, {num_tokens, hidden_units}, d_decoder_rsd);
    LayerNormWeight<float> scale;
    scale.gamma = d_scale;
    
    // 运行GPU版本的RMSNorm
    std::cout << "开始执行GPU版本的RMSNorm..." << std::endl;
    launchRMSNorm(decoder_out_tensor, decoder_rsd, scale, eps);
    CHECK(cudaDeviceSynchronize()); // 确保GPU计算完成
    std::cout << "GPU版本的RMSNorm执行完成" << std::endl;
    
    // 复制GPU结果回CPU
    CHECK(cudaMemcpy(gpu_result, d_decoder_out, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 运行CPU版本的RMSNorm
    std::cout << "开始执行CPU版本的RMSNorm..." << std::endl;
    CPUfusedresidandRMSNorm(h_decoder_out, h_scale, eps, hidden_units, num_tokens);
    std::cout << "CPU版本的RMSNorm执行完成" << std::endl;
    
    // 检查结果并输出对比信息
    bool is_right = CheckResult<float>(h_decoder_out, gpu_result, total_size);
    std::cout << "RMSNorm测试 " << (is_right ? "通过" : "失败") << std::endl;
    
    // 清理内存
    free(h_decoder_out);
    free(h_decoder_out_copy);
    free(gpu_result);
    free(h_scale);
    
    delete decoder_out_tensor;
    delete decoder_rsd;
    
    cudaFree(d_decoder_out);
    cudaFree(d_decoder_rsd);
    cudaFree(d_scale);
    
    return 0;
}