#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <iostream>
#include "src/kernels/add_residual.h"

#include <stdio.h>
// this kernel's CPU implementation is absolutely right.
// But when you are implementing LLMs inference on CPU, I dont recommend to reuse the CPU kernel, because its performance is bad //这里可以使用AVX512 以及openmp来并行cpu函数以提升效率
// Usage:
// `./test_residual` to test fp32 GPU kernel
// `./test_residual fp16` to test fp16 GPU kernel
// `./test_residual int8` to test int8 GPU kernel

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
void CPUresidual(T* h_residual, T* h_decoder_out, int hidden_units, int num_tokens) {
    for(int b = 0; b < num_tokens; b++) {
        for (int i = 0; i < hidden_units; i++) {
            h_decoder_out[b * hidden_units + i] += h_residual[b * hidden_units + i];
        }
    }
}

template<typename T>
bool CheckResult(T* CPUoutput, T* GPUoutput, int output_size) {
    float tolerance = std::is_same<T, int8_t>::value ? 1e-1f : 
                      (std::is_same<T, half>::value ? 1e-3f : 1e-6f);
    for(int i = 0; i < output_size; i++) {
        if(fabs((float)CPUoutput[i] - (float)GPUoutput[i]) > tolerance){
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, (float)CPUoutput[i], (float)GPUoutput[i]);
            return false;
        }
    }
    return true;
}

template<typename T>
void test_residual_kernel() {
    const int num_tokens = 16;
    const int hidden_units = 4096;
    const int total_size = num_tokens * hidden_units;
    
    printf("Testing %s residual kernel with %d tokens, %d hidden units\n", 
           typeid(T).name(), num_tokens, hidden_units);
    
    T* h_residual;
    T* d_residual;
    h_residual = (T*)malloc(sizeof(T) * total_size);
    cudaMalloc((void**)&d_residual, sizeof(T) * total_size);
    
    for(int i = 0; i < total_size; i++) { 
        if (std::is_same<T, int8_t>::value) {
            h_residual[i] = (T)(i % 10 - 5); // Range [-5, 4] for INT8
        } else {
            h_residual[i] = (T)(i % 2 + 1);  // 1, 2, 1, 2... pattern
        }
    }

    T* h_decoder_out = (T*) malloc(sizeof(T) * total_size);
    T* decoder_out = (T*) malloc(sizeof(T) * total_size);
    T* d_decoder_out;
    cudaMalloc((void**)&d_decoder_out, sizeof(T) * total_size);
    
    for(int i = 0; i < total_size; i++) { 
        if (std::is_same<T, int8_t>::value) {
            h_decoder_out[i] = (T)(i % 8 - 4); // Range [-4, 3] for INT8
        } else {
            h_decoder_out[i] = (T)(i % 2 + 1);
        }
    }

    CHECK(cudaMemcpy(d_residual, h_residual, sizeof(T) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(T) * total_size, cudaMemcpyHostToDevice));
    
    DataType type = getTensorType<T>();
    TensorWrapper<T>* decoder_out_tensor = new TensorWrapper<T>(Device::GPU, 
                                                                        type,
                                                                        {num_tokens, hidden_units}, 
                                                                        d_decoder_out);
    TensorWrapper<T>* residual_tensor = new TensorWrapper<T>(Device::GPU, 
                                                                        type,
                                                                        {num_tokens, hidden_units}, 
                                                                        d_residual);                                                                        
    std::cout << "before launch kernel" << std::endl;
    launchAddResidual(residual_tensor, decoder_out_tensor);
    std::cout << "after launch kernel" << std::endl;
    std::cout << "cuda memcpy device to host" << std::endl;
    
    CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(T) * total_size, cudaMemcpyDeviceToHost));
    T* CPUout = (T*) malloc(sizeof(T) * total_size);
    for(int i = 0; i < total_size; i++){
        if (std::is_same<T, int8_t>::value) {
            CPUout[i] = (T)(i % 8 - 4);
        } else {
            CPUout[i] = (T)(i % 2 + 1);
        }
    }
    CPUresidual(h_residual, CPUout, hidden_units, num_tokens);
    bool is_right = CheckResult(CPUout, decoder_out, total_size);
    
    std::cout << "before free" << std::endl;
    if (is_right) {
        std::cout << "AddResidual kernel passed" << std::endl;
    } else {
        std::cout << "AddResidual kernel failed" << std::endl;
    }
    
    free(h_residual);
    free(h_decoder_out);
    free(CPUout);
    free(decoder_out);
    cudaFree(d_residual);
    cudaFree(d_decoder_out);
    delete decoder_out_tensor;
    delete residual_tensor;
}

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "fp16") {
            test_residual_kernel<half>();
        } else if (arg == "int8") {
            test_residual_kernel<int8_t>();
        } else {
            test_residual_kernel<float>();
        }
    } else {
        test_residual_kernel<float>();
    }
    
    return 0;
}
