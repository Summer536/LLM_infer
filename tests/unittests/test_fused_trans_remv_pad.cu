#include "src/kernels/fused_transpose_and_remv_pad.h"
#include <iostream>
#include <string>
// [b,h,s,d]=>[b,s,h,d]=>[num tokens,h,d]
// padding_offset.shape = [num_tokens]
// Usage:
// `./test_fused_trans_remv_pad` to test fp32 kernel
// `./test_fused_trans_remv_pad fp16` to test fp16 kernel
// `./test_fused_trans_remv_pad int8` to test int8 kernel

template<typename T>
void test_fused_transpose_kernel() {
    const int batch_size = 2;
    const int head_num = 2;
    const int max_seq_len = 4;
    const int head_size = 2;
    const int num_tokens = 5;
    const int in_size = batch_size * head_num * max_seq_len * head_size;
    const int out_size = num_tokens * head_num * head_size;
    
    printf("Testing %s fused transpose and remove padding kernel\n", typeid(T).name());
    
    T* h_in;
    T* d_in;
    h_in = (T*)malloc(sizeof(T) * in_size);
    cudaMalloc((void**)&d_in, sizeof(T) * in_size);
    T* h_out;
    T* d_out;
    h_out = (T*)malloc(sizeof(T) * out_size);
    cudaMalloc((void**)&d_out, sizeof(T) * out_size);
    int* h_padding_offset;
    int* d_padding_offset;
    h_padding_offset = (int*)malloc(sizeof(int) * num_tokens);
    cudaMalloc((void**)&d_padding_offset, sizeof(int) * num_tokens);

    //1st seqlen: 2, due to 1st seq, so its padding offset are all 0
    //2nd seqlen: 3, so its padding offset are all 4-2=2
    for(int i = 0; i < in_size; i++) {
        if (std::is_same<T, int8_t>::value) {
            h_in[i] = (T)(i % 20 - 10); // Range [-10, 9] for INT8
        } else {
            h_in[i] = (T)i;
        }
    }
    for(int i = 0; i < 2; i++) {
       h_padding_offset[i] = 0;
    } 
    h_padding_offset[2] = 2;  
    h_padding_offset[3] = 2;
    h_padding_offset[4] = 2;

    cudaMemcpy(d_in, h_in, sizeof(T) * in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_padding_offset, h_padding_offset, sizeof(int) * num_tokens, cudaMemcpyHostToDevice);

    DataType type = getTensorType<T>(); 
    DataType type_pad = getTensorType<int>(); 
    TensorWrapper<T>* in = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_seq_len, head_size}, d_in);
    TensorWrapper<int>* in_pad = new TensorWrapper<int>(Device::GPU, type_pad, {num_tokens}, d_padding_offset);
    TensorWrapper<T>* out = new TensorWrapper<T>(Device::GPU, type, {num_tokens, head_num, head_size}, d_out);
    
    std::cout << "before launch transpose kernel" << std::endl;
    launchTransposeOutRemovePadding(in, in_pad, out);
    std::cout << "after launch transpose kernel" << std::endl;
    std::cout << "cuda memcpy device to host" << std::endl;
    
    cudaMemcpy(h_out, out->data, sizeof(T) * out_size, cudaMemcpyDeviceToHost);
    
    printf("Results (showing first 16 elements):\n");
    for(int i = 0; i < out_size && i < 16; i++) {
        printf("after trans and remv pad, out[%d] = %f\n", i, (float)h_out[i]);
    }
    if (out_size > 16) printf("... (showing first 16 results only)\n");
    printf("Test completed successfully\n");
    
    free(h_in);
    free(h_out);
    free(h_padding_offset);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_padding_offset);
    delete in;
    delete in_pad;
    delete out;
}

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "fp16") {
            test_fused_transpose_kernel<half>();
        } else if (arg == "int8") {
            test_fused_transpose_kernel<int8_t>();
        } else {
            test_fused_transpose_kernel<float>();
        }
    } else {
        test_fused_transpose_kernel<float>();
    }
    
    return 0;
}