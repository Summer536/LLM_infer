#include <iostream>
#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "src/kernels/cal_paddingoffset.h"

int main(){
    const int batch_size = 3;
    const int max_seq_len = 5;
    int *h_seq_lens;
    int *d_seq_lens;
    h_seq_lens = (int*)malloc(batch_size * sizeof(int));
    cudaMalloc((void**)&d_seq_lens, batch_size * sizeof(int));

    int *h_cum_seqlens;
    int *d_cum_seqlens;
    h_cum_seqlens = (int*)malloc((batch_size + 1) * sizeof(int));
    cudaMalloc((void**)&d_cum_seqlens, (batch_size + 1) * sizeof(int));

    int *h_padding_offset;
    int *d_padding_offset;
    h_padding_offset = (int*)malloc(batch_size * max_seq_len * sizeof(int));
    cudaMalloc((void**)&d_padding_offset, batch_size * max_seq_len * sizeof(int));

    for (int i = 0; i < batch_size; i++){
        h_seq_lens[i] = batch_size;
    }
    cudaMemcpy(d_seq_lens, h_seq_lens, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    DataType type_int = getTensorType<int>();
    TensorWrapper<int> *padding_offset = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, max_seq_len}, d_padding_offset);
    TensorWrapper<int> *cum_seqlens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size + 1}, d_cum_seqlens);
    TensorWrapper<int> *input_lengths = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_seq_lens);

    launchCalPaddingoffset(padding_offset, cum_seqlens, input_lengths);

    cudaMemcpy(h_padding_offset, d_padding_offset, batch_size * max_seq_len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cum_seqlens, d_cum_seqlens, (batch_size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; i++){
        printf("padding_offset = %d\n", h_padding_offset[i]);
    }
    for (int i = 0; i < batch_size + 1; i++){
        printf("cum_seqlens = %d\n", h_cum_seqlens[i]);
    }

    free(h_seq_lens);
    free(h_cum_seqlens);
    free(h_padding_offset);
    cudaFree(d_seq_lens);
    cudaFree(d_cum_seqlens);
    cudaFree(d_padding_offset);

    return 0;
}





