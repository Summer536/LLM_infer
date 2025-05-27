#include <algorithm>   
#include <iostream>    
#include <math.h>      
#include <stdlib.h>    
#include <string>      
#include <vector>

#include "src/kernels/build_causal_mask.h"

// Usage:
// `./test_causal_mask` to test fp32 GPU kernel
// `./test_causal_mask fp16` to test fp16 GPU kernel  
// `./test_causal_mask int8` to test int8 GPU kernel

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
void CPUbuildCausalMasks(T *mask, const int *q_lens, const int *k_lens, int max_q_len, int max_k_len, int batch_size){
    for(int b = 0; b < batch_size; b++){
        int start = b * max_q_len * max_k_len;
        int q = q_lens[b];
        int k = k_lens[b];
        for(int i = 0; i < max_q_len; i++) {
            for(int j = 0; j < max_k_len; j++) {
                if(j <= i + (k - q) && i < q && j < k) {
                    mask[start + i * max_k_len + j] = (T)1.0f;
                } else {
                    mask[start + i * max_k_len + j] = (T)0.0f;   
                }
            }
        }
    }
}

template<typename T>
bool CheckResult(T *CPUres, T *GPUres, int size){
    float tolerance = std::is_same<T, int8_t>::value ? 1e-1f : 
                      (std::is_same<T, half>::value ? 1e-3f : 1e-5f);
    for(int i = 0; i < size; i++){
        if(fabs((float)CPUres[i] - (float)GPUres[i]) > tolerance){
            printf("the %dth res is wrong, CPU mask = %f, GPU mask = %f\n", i, (float)CPUres[i], (float)GPUres[i]);
            return false;
        }
    }
    return true;
}

template<typename T>
void test_causal_mask_kernel(){
    const int batch_size = 1;
    const int max_q_len = 5;
    const int max_k_len = 5;
    const int mask_size = batch_size * max_q_len * max_k_len;
    
    printf("Testing %s causal mask kernel\n", typeid(T).name());
    
    int *h_q_lens = (int*)malloc(batch_size * sizeof(int));
    int *d_q_lens;
    cudaMalloc((void**)&d_q_lens, batch_size * sizeof(int));

    int *h_k_lens = (int*)malloc(batch_size * sizeof(int));
    int *d_k_lens;
    cudaMalloc((void**)&d_k_lens, batch_size * sizeof(int));

    T *h_mask = (T*)malloc(mask_size * sizeof(T));
    T *d_mask;
    cudaMalloc((void**)&d_mask, mask_size * sizeof(T));

    for(int i = 0; i < batch_size; i++) {
       h_q_lens[i] = 3;
    }
    for(int i = 0; i < batch_size; i++) {
       h_k_lens[i] = 3;
    }

    cudaMemcpy(d_q_lens, h_q_lens, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_lens, h_k_lens, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    DataType type = getTensorType<T>();
    DataType type_int = getTensorType<int>();
    TensorWrapper<T> *mask = new TensorWrapper<T>(Device::GPU, type, {batch_size, max_q_len, max_k_len}, d_mask);
    TensorWrapper<int> *q_lens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_q_lens);
    TensorWrapper<int> *k_lens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_k_lens);

    launchBuildCausalMasks(mask, q_lens, k_lens);
    CHECK(cudaMemcpy(h_mask, d_mask, mask_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    T *CPUmask = (T*)malloc(mask_size * sizeof(T));
    CPUbuildCausalMasks(CPUmask, h_q_lens, h_k_lens, max_q_len, max_k_len, batch_size);
    
    if (CheckResult(CPUmask, h_mask, mask_size)){
        printf("test causal mask success\n");
    } else {
        printf("test causal mask failed\n");
    }

    free(h_q_lens);
    free(h_k_lens);
    free(h_mask);
    free(CPUmask);
    cudaFree(d_q_lens);
    cudaFree(d_k_lens);
    cudaFree(d_mask);
    delete mask;
    delete q_lens;
    delete k_lens;
}

int main(int argc, char** argv){
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "fp16") {
            test_causal_mask_kernel<half>();
        } else if (arg == "int8") {
            test_causal_mask_kernel<int8_t>();
        } else {
            test_causal_mask_kernel<float>();
        }
    } else {
        test_causal_mask_kernel<float>();
    }
    
    return 0;
}