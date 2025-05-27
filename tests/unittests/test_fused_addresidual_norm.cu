#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <iostream>
#include "src/kernels/fused_addresidual_norm.h"

#include <stdio.h>

// Usage:
// `./test_fused_addresidual_norm` to test fp32 GPU kernel
// `./test_fused_addresidual_norm fp16` to test fp16 GPU kernel
// `./test_fused_addresidual_norm int8` to test int8 GPU kernel
// this kernel's CPU implementation is absolutely right.
// when you are implementing LLMs inference on CPU, you can reuse the CPU kernel

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

template<typename T>
void CPUfusedresidandRMSNorm(T* h_residual, T* h_decoder_out, T* h_bias, 
                                    T* h_scale, float eps, int hidden_units, int num_tokens) {
    for(int b = 0; b < num_tokens; b++) {
        float inv_fenmu = 0.0f;
        float mean = 0.0f;
        float sum = 0.0f;
        
        // Add residual and bias first, then compute RMSNorm
        for (int i = 0; i < hidden_units; i++) {
            float input = (float)h_decoder_out[b * hidden_units + i] +
                         (float)h_residual[b * hidden_units + i] + 
                         (float)h_bias[i];
            h_decoder_out[b * hidden_units + i] = (T)input;
            sum += input * input;
        }
        
        mean = sum / hidden_units;
        inv_fenmu = rsqrt(mean + eps);
        
        for (int i = 0; i < hidden_units; i++) {
            float normalized = (float)h_decoder_out[b * hidden_units + i] * inv_fenmu * (float)h_scale[i];
            h_decoder_out[b * hidden_units + i] = (T)normalized;
        }
    }
}

template<typename T>
bool CheckResult(T* CPUoutput, T* GPUoutput, int output_size) {
    float tolerance = std::is_same<T, int8_t>::value ? 1e-1f : 
                      (std::is_same<T, half>::value ? 1e-2f : 1e-6f);
    for(int i = 0; i < output_size; i++) {
        if(fabs((float)CPUoutput[i] - (float)GPUoutput[i]) > tolerance){
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, (float)CPUoutput[i], (float)GPUoutput[i]);
            return false;
        }
    }
    return true;
}

template<typename T>
void test_fused_norm_kernel() {
    const int num_tokens = 2;
    const int hidden_units = 32;
    const int total_size = num_tokens * hidden_units;
    float eps = 0.5f;
    
    printf("Testing %s fused add residual norm kernel\n", typeid(T).name());
    
    T* h_residual;
    T* d_residual;
    h_residual = (T*)malloc(sizeof(T) * total_size);
    cudaMalloc((void**)&d_residual, sizeof(T) * total_size);
    
    for(int i = 0; i < total_size; i++) { 
        if (std::is_same<T, int8_t>::value) {
            h_residual[i] = (T)(i % 6 - 3); // Range [-3, 2] for INT8
        } else {
            h_residual[i] = (T)0.0f;
        }
    }

    T* h_decoder_out = (T*) malloc(sizeof(T) * total_size);
    T* decoder_out = (T*) malloc(sizeof(T) * total_size);
    T* d_decoder_out;
    cudaMalloc((void**)&d_decoder_out, sizeof(T) * total_size);
    
    for(int i = 0; i < total_size; i++) { 
        if (std::is_same<T, int8_t>::value) {
            h_decoder_out[i] = (T)(i % 4 - 2); // Range [-2, 1] for INT8
        } else {
            h_decoder_out[i] = (T)1.0f;
        }
    }
    
    //bias
    T* h_bias = (T*) malloc(sizeof(T) * hidden_units);
    T* d_bias;
    cudaMalloc((void**)&d_bias, sizeof(T) * hidden_units);
    for(int i = 0; i < hidden_units; i++) { 
        h_bias[i] = (T)0.0f;
    }
    
    //rmsnorm weights
    T* h_scale = (T*) malloc(sizeof(T) * hidden_units);
    T* d_scale;
    cudaMalloc((void**)&d_scale, sizeof(T) * hidden_units);
    for(int i = 0; i < hidden_units; i++) { 
        h_scale[i] = (T)1.0f;
    }

    CHECK(cudaMemcpy(d_residual, h_residual, sizeof(T) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(T) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, h_bias, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scale, h_scale, sizeof(T) * hidden_units, cudaMemcpyHostToDevice));
    
    DataType type = getTensorType<T>();
    TensorWrapper<T>* decoder_out_tensor = new TensorWrapper<T>(Device::GPU, 
                                                                        type,
                                                                        {num_tokens, hidden_units}, 
                                                                        d_decoder_out);
    TensorWrapper<T>* residual_tensor = new TensorWrapper<T>(Device::GPU, 
                                                                        type,
                                                                        {num_tokens, hidden_units}, 
                                                                        d_residual);                                                                        
    BaseWeight<T> norm;
    
    std::cout << "before launch kernel" << std::endl;
    launchFusedAddBiasResidualRMSNorm(residual_tensor, 
                                    decoder_out_tensor, 
                                    norm,
                                    d_scale,
                                    eps);
    std::cout << "after launch kernel" << std::endl;
    std::cout << "cuda memcpy device to host" << std::endl;
    
    CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(T) * total_size, cudaMemcpyDeviceToHost));
    T* CPUout = (T*) malloc(sizeof(T) * total_size);
    for(int i = 0; i < total_size; i++){
        if (std::is_same<T, int8_t>::value) {
            CPUout[i] = (T)(i % 4 - 2);
        } else {
            CPUout[i] = (T)1.0f;
        }
    }
    CPUfusedresidandRMSNorm(h_residual, CPUout, h_bias, 
                h_scale, eps, hidden_units, num_tokens);
    bool is_right = CheckResult(CPUout, decoder_out, total_size);
    
    std::cout << "before free" << std::endl;
    if (is_right) {
        std::cout << "fused addres and rmsnorm passed" << std::endl;
    } else {
        std::cout << "fused addres and rmsnorm failed" << std::endl;
    }
    
    free(h_residual);
    free(h_decoder_out);
    free(h_bias);
    free(h_scale);
    free(CPUout);
    free(decoder_out);
    cudaFree(d_residual);
    cudaFree(d_decoder_out);
    cudaFree(d_bias);
    cudaFree(d_scale);
    delete decoder_out_tensor;
    delete residual_tensor;
}

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "fp16") {
            test_fused_norm_kernel<half>();
        } else if (arg == "int8") {
            test_fused_norm_kernel<int8_t>();
        } else {
            test_fused_norm_kernel<float>();
        }
    } else {
        test_fused_norm_kernel<float>();
    }
    
    return 0;
}
