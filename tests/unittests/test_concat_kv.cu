#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include <cuda.h>
#include "src/kernels/concat_past_kv.h"
// (RussWong)note:
// there is no concat kv cpu kernel implementation now
// we compare the kernel correctnesss by eyes and result print infos
// Usage:
// `./test_concat_kv` to test fp32 GPU kernel
// `./test_concat_kv fp16` to test fp16 GPU kernel  
// `./test_concat_kv int8` to test int8 GPU kernel

template<typename T>
void test_concat_kv_kernel()
{
    const int batch_size = 1;
    const int max_q_len = 16;
    const int max_seq_len = 32;
    const int head_size = 8;
    const int kv_head_num = 2;
    const int kv_size = 1 * batch_size * max_q_len * kv_head_num * head_size;
    const int layer_offset = 1 * batch_size * max_seq_len * kv_head_num * head_size;
    const int kvcache_size = layer_offset;
    
    printf("Testing %s concat KV kernel\n", typeid(T).name());

    T *h_k_src;
    T *d_k_src;
    h_k_src = (T *)malloc(sizeof(T) * kv_size);
    cudaMalloc((void **)&d_k_src, sizeof(T) * kv_size);

    T *h_v_src;
    T *d_v_src;
    h_v_src = (T *)malloc(sizeof(T) * kv_size);
    cudaMalloc((void **)&d_v_src, sizeof(T) * kv_size);

    int *cur_query_length = (int *)malloc(sizeof(int) * batch_size);
    int *history_length = (int *)malloc(sizeof(int) * batch_size);
    int *dcur_query_length;
    int *dhistory_length;
    cudaMalloc((void **)&dcur_query_length, sizeof(int) * batch_size);
    cudaMalloc((void **)&dhistory_length, sizeof(int) * batch_size);

    T *h_k_dst = (T *)malloc(sizeof(T) * kvcache_size);
    T *h_v_dst = (T *)malloc(sizeof(T) * kvcache_size);
    T *d_k_dst;
    T *d_v_dst;
    cudaMalloc((void **)&d_k_dst, sizeof(T) * kvcache_size);
    cudaMalloc((void **)&d_v_dst, sizeof(T) * kvcache_size);
    
    int *h_layer_id = (int *)malloc(sizeof(int) * batch_size);

    for (int i = 0; i < kv_size; i++)
    {
        if (std::is_same<T, int8_t>::value) {
            h_k_src[i] = (T)(i % 10 - 5);
            h_v_src[i] = (T)(i % 8 - 4);
        } else {
            h_k_src[i] = (T)1.0f;
            h_v_src[i] = (T)1.0f;
        }
    }
    for (int i = 0; i < batch_size; i++)
    {
        cur_query_length[i] = 16;
        history_length[i] = 1;
        h_layer_id[i] = 0;
    }
    cudaMemcpy(d_v_src, h_v_src, sizeof(T) * kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_src, h_k_src, sizeof(T) * kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcur_query_length, cur_query_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dhistory_length, history_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice);

    DataType type = getTensorType<T>();
    DataType type_int = getTensorType<int>();
    TensorWrapper<T> *in_ksrc = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size}, d_k_src);
    TensorWrapper<T> *in_vsrc = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size}, d_v_src);
    TensorWrapper<int> *layer_id = new TensorWrapper<int>(Device::CPU, type_int, {batch_size}, h_layer_id); //为什么只有这个是CPU，因为没有必要拷贝到GPU上，不会参与kernel的具体计算，因此设置到了cpu上
    TensorWrapper<int> *cur_q_len = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, dcur_query_length);
    TensorWrapper<int> *history_len = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, dhistory_length);
    TensorWrapper<T> *out_kdst = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_seq_len, head_size}, d_k_dst);
    TensorWrapper<T> *out_vdst = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_seq_len, head_size}, d_v_dst);
    
    std::cout << "before launch kernel" << std::endl;
    launchConcatKVCache(in_ksrc, in_vsrc, layer_id, cur_q_len, history_len, out_kdst, out_vdst);
    std::cout << "after launch kernel" << std::endl;
    
    cudaMemcpy(h_v_dst, d_v_dst, sizeof(T) * kvcache_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_dst, d_k_dst, sizeof(T) * kvcache_size, cudaMemcpyDeviceToHost);
    std::cout << "cuda memcpy device to host" << std::endl;
    
    // Print some results for verification
    for (int i = batch_size * (1) * kv_head_num * head_size; i < batch_size * max_seq_len * kv_head_num * head_size && i < batch_size * (1) * kv_head_num * head_size + 10; i++)
    {
        printf("index = %d\n", i);
        printf("res k = %f\n", (float)h_k_dst[i]);
        printf("res v = %f\n", (float)h_v_dst[i]);
        printf("===============\n");
    }
    
    printf("Concat KV test completed\n");
    
    free(h_k_src);
    free(h_v_src);
    free(h_k_dst);
    free(h_v_dst);
    free(cur_query_length);
    free(history_length);
    free(h_layer_id);
    cudaFree(d_k_src);
    cudaFree(d_v_src);
    cudaFree(d_k_dst);
    cudaFree(d_v_dst);
    cudaFree(dcur_query_length);
    cudaFree(dhistory_length);
    
    delete in_ksrc;
    delete in_vsrc;
    delete layer_id;
    delete cur_q_len;
    delete history_len;
    delete out_kdst;
    delete out_vdst;
}

int main(int argc, char** argv)
{
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "fp16") {
            test_concat_kv_kernel<half>();
        } else if (arg == "int8") {
            test_concat_kv_kernel<int8_t>();
        } else {
            test_concat_kv_kernel<float>();
        }
    } else {
        test_concat_kv_kernel<float>();
    }
    
    return 0;
}
