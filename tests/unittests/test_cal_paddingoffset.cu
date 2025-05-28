#include <iostream>
#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cassert>

#include "src/kernels/cal_paddingoffset.h"

void printResults(const char* name, int* data, int size) {
    std::cout << name << ": ";
    for (int i = 0; i < size; i++) {
        std::cout << data[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << std::endl;
}

void testCalPaddingOffset(const std::vector<int>& seq_lens, int max_seq_len) {
    const int batch_size = seq_lens.size();
    
    std::cout << "\n=== Testing CalPaddingOffset ===" << std::endl;
    std::cout << "Batch size: " << batch_size << ", Max seq len: " << max_seq_len << std::endl;
    std::cout << "Input seq lengths: ";
    for (int i = 0; i < batch_size; i++) {
        std::cout << seq_lens[i];
        if (i < batch_size - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Allocate host memory
    int *h_seq_lens = (int*)malloc(batch_size * sizeof(int));
    int *h_cum_seqlens = (int*)malloc((batch_size + 1) * sizeof(int));
    int *h_padding_offset = (int*)malloc(batch_size * max_seq_len * sizeof(int));
    
    // Allocate device memory
    int *d_seq_lens, *d_cum_seqlens, *d_padding_offset;
    cudaMalloc((void**)&d_seq_lens, batch_size * sizeof(int));
    cudaMalloc((void**)&d_cum_seqlens, (batch_size + 1) * sizeof(int));
    cudaMalloc((void**)&d_padding_offset, batch_size * max_seq_len * sizeof(int));

    // Initialize input data
    for (int i = 0; i < batch_size; i++) {
        h_seq_lens[i] = seq_lens[i];
    }
    
    // Copy to device
    cudaMemcpy(d_seq_lens, h_seq_lens, batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Create tensor wrappers
    DataType type_int = getTensorType<int>();
    TensorWrapper<int> *padding_offset = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, max_seq_len}, d_padding_offset);
    TensorWrapper<int> *cum_seqlens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size + 1}, d_cum_seqlens);
    TensorWrapper<int> *input_lengths = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_seq_lens);

    // Launch kernel
    launchCalPaddingoffset(padding_offset, cum_seqlens, input_lengths);
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Wait for kernel completion
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_padding_offset, d_padding_offset, batch_size * max_seq_len * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cum_seqlens, d_cum_seqlens, (batch_size + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "\nResults:" << std::endl;
    printResults("Cumulative seq lengths", h_cum_seqlens, batch_size + 1);
    
    std::cout << "Padding offsets:" << std::endl;
    for (int b = 0; b < batch_size; b++) {
        std::cout << "  Batch " << b << " (len=" << seq_lens[b] << "): ";
        for (int i = 0; i < seq_lens[b]; i++) {
            std::cout << h_padding_offset[b * max_seq_len + i];
            if (i < seq_lens[b] - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    // Verify correctness
    bool correct = true;
    
    // Check cumulative sequence lengths
    int expected_total = 0;
    for (int i = 0; i < batch_size; i++) {
        if (h_cum_seqlens[i] != expected_total) {
            std::cerr << "Error: cum_seqlens[" << i << "] = " << h_cum_seqlens[i] 
                      << ", expected " << expected_total << std::endl;
            correct = false;
        }
        expected_total += seq_lens[i];
    }
    if (h_cum_seqlens[batch_size] != expected_total) {
        std::cerr << "Error: cum_seqlens[" << batch_size << "] = " << h_cum_seqlens[batch_size] 
                  << ", expected " << expected_total << std::endl;
        correct = false;
    }
    
    // Check padding offsets
    int expected_offset = 0;
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < seq_lens[b]; i++) {
            int offset_idx = b * max_seq_len + i;
            if (h_padding_offset[offset_idx] != expected_offset) {
                std::cerr << "Error: padding_offset[" << b << "][" << i << "] = " 
                          << h_padding_offset[offset_idx] << ", expected " << expected_offset << std::endl;
                correct = false;
            }
        }
        expected_offset += max_seq_len - seq_lens[b];
    }
    
    if (correct) {
        std::cout << "✓ Test PASSED" << std::endl;
    } else {
        std::cout << "✗ Test FAILED" << std::endl;
    }

    // Cleanup
    free(h_seq_lens);
    free(h_cum_seqlens);
    free(h_padding_offset);
    cudaFree(d_seq_lens);
    cudaFree(d_cum_seqlens);
    cudaFree(d_padding_offset);
    
    delete padding_offset;
    delete cum_seqlens;
    delete input_lengths;
}

int main() {
    std::cout << "Testing CalPaddingOffset kernel..." << std::endl;
    
    // Test case 1: All sequences have the same length
    testCalPaddingOffset({3, 3, 3}, 5);
    
    // Test case 2: Different sequence lengths
    testCalPaddingOffset({2, 4, 1}, 5);
    
    // Test case 3: Single sequence
    testCalPaddingOffset({3}, 5);
    
    // Test case 4: Sequences at max length
    testCalPaddingOffset({5, 5}, 5);
    
    // Test case 5: Mix of short and long sequences  
    testCalPaddingOffset({1, 5, 3, 2}, 6);
    
    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}





