#include <algorithm>   
#include <iostream>    
#include <math.h>      
#include <stdlib.h>    
#include <string>      
#include <vector>

#include "src/kernels/build_causal_mask.h"

void CPUbuildCausalMasks(float *mask, const int *q_lens, const int *k_lens, int max_q_len, int max_k_len, int batch_size){
    for(int b = 0; b < batch_size; b++){
        int start = b * max_q_len * max_k_len;
        int q = q_lens[b];
        int k = k_lens[b];
        for(int i = 0; i < max_q_len; i++) {
            for(int j = 0; j < max_k_len; j++) {
                if(j <= i + (k - q) && i < q && j < k) {
                    mask[start + i * max_k_len + j] = 1.0f;
                } else {
                    mask[start + i * max_k_len + j] = 0.0f;   
                }
            }
        }
    }
}

bool CheckResult(float *CPUres, float *GPUres, int size){
    for(int i = 0; i < size; i++){
        if(fabs(CPUres[i] - GPUres[i]) > 1e-5){
            printf("the %dth res is wrong, CPU mask = %f, GPU mask = %f\n", i, CPUres[i], GPUres[i]);
            return false;
        }
    }
    return true;
}

int main(){
    const int batch_size = 1;
    const int max_q_len = 5;
    const int max_k_len = 5;
    const int mask_size = batch_size * max_q_len * max_k_len;
    int *h_q_lens = (int*)malloc(batch_size * sizeof(int));
    int *d_q_lens;
    cudaMalloc((void**)&d_q_lens, batch_size * sizeof(int));

    int *h_k_lens = (int*)malloc(batch_size * sizeof(int));
    int *d_k_lens;
    cudaMalloc((void**)&d_k_lens, batch_size * sizeof(int));

    float *h_mask = (float*)malloc(mask_size * sizeof(float));
    float *d_mask;
    cudaMalloc((void**)&d_mask, mask_size * sizeof(float));

    for(int i = 0; i < batch_size; i++) {
       h_q_lens[i] = 3;
    }
    for(int i = 0; i < batch_size; i++) {
       h_k_lens[i] = 3;
    }

    cudaMemcpy(d_q_lens, h_q_lens, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_lens, h_k_lens, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    DataType type_float = getTensorType<float>();
    DataType type_int = getTensorType<int>();
    TensorWrapper<float> *mask = new TensorWrapper<float>(Device::GPU, type_float, {batch_size, max_q_len, max_k_len}, d_mask);
    TensorWrapper<int> *q_lens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_q_lens);
    TensorWrapper<int> *k_lens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_k_lens);

    launchBuildCausalMasks(mask, q_lens, k_lens);
    CHECK(cudaMemcpy(h_mask, d_mask, mask_size * sizeof(float), cudaMemcpyDeviceToHost));
    float *CPUmask = (float*)malloc(mask_size * sizeof(float));
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
}