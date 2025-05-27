#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <random>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/kernels/input_embedding.h"

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

void cpuEmbedding(const int* input_ids, float* output, float* embed_table, const int max_context_token_num, const int hidden_size, const int vocab_size) {
    for (int i = 0; i < max_context_token_num; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            output[j + i * hidden_size] = embed_table[j + input_ids[i] * hidden_size];
        }
    }
}

bool checkResults(float* h_output, float* d_output, const int output_size) {
    float* d_output_cpu = (float*) malloc(output_size * sizeof(float));
    CHECK(cudaMemcpy(d_output_cpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < output_size; ++i) {
        if (fabs(d_output_cpu[i] - h_output[i]) > 1e5) {
            std::cout << "Dev : ";
            for (int j = max(0, i - 10); j < min(output_size, i + 10); ++j) {
                std::cout << d_output_cpu[i];
            }
            std::cout << std::endl;
            std::cout << "Cpu : ";
            for (int j = max(0, i - 10); j < min(output_size, i + 10); ++j) {
                std::cout << h_output[i];
            }
            std::cout << std::endl;
            free(d_output_cpu);
            return false;
        }
    }
    free(d_output_cpu);
    return true;
}

template<typename T>
void test_input_embedding_kernel() {
    const int max_context_token_num = 64;
    const int hidden_size = 4096;
    const int vocab_size = 32000;
    const int input_size = max_context_token_num;
    const int table_size = vocab_size * hidden_size;
    const int output_size = max_context_token_num * hidden_size;

    printf("Testing %s input embedding kernel\n", typeid(T).name());

    int* h_input = (int*) malloc(input_size * sizeof(int));
    T* h_table = (T*) malloc(table_size * sizeof(T));
    T* h_output = (T*) malloc(output_size * sizeof(T));

    std::cout << "init memory on host" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_int(0, vocab_size - 1);

    for (int i = 0; i < max_context_token_num; ++i) {
        h_input[i] = dis_int(gen);
        if (i < 5) printf("h_input[%d] = %d\n", i, h_input[i]);
    }
    
    // Initialize table with simple pattern for testing
    for (int i = 0; i < table_size; ++i) {
        if (std::is_same<T, int8_t>::value) {
            h_table[i] = (T)((i / hidden_size) % 20 - 10); // Range [-10, 9] for INT8
        } else {
            h_table[i] = (T)(i / hidden_size);
        }
    }

    int* d_input;
    T *d_table, *d_output;
    cudaMalloc((void**)&d_input, input_size * sizeof(int));
    cudaMalloc((void**)&d_table, table_size * sizeof(T));
    cudaMalloc((void**)&d_output, output_size * sizeof(T));
    std::cout << "init memory on device" << std::endl;

    CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_table, h_table, table_size * sizeof(T), cudaMemcpyHostToDevice));
    std::cout << "copy to device" << std::endl;

    DataType type = getTensorType<T>();
    DataType type_int = getTensorType<int>();
    TensorWrapper<int>* input_ids = new TensorWrapper<int>(Device::GPU, type_int, {max_context_token_num}, d_input);
    TensorWrapper<T>* output = new TensorWrapper<T>(Device::GPU, type, {max_context_token_num, hidden_size}, d_output);
    EmbeddingWeight<T> emb_table;
    emb_table.data = d_table;

    launchInputEmbedding(input_ids, output, &emb_table);
    CHECK(cudaMemcpy(h_output, output->data, output_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    std::cout << "printf h_output for check (first 5 embeddings):" << std::endl;
    for (int i = 0; i < 5 && i < max_context_token_num; i++){
        std::cout << "embedding[" << i << "][0] = " << (float)h_output[i * hidden_size] << std::endl;
    }
    
    printf("Test completed successfully\n");

    cudaFree(d_output);
    cudaFree(d_table);
    cudaFree(d_input);
    free(h_output);
    free(h_table);
    free(h_input);
    delete input_ids;
    delete output;
}

int main(int argc, char *argv[]) {
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "fp16") {
            test_input_embedding_kernel<half>();
        } else if (arg == "int8") {
            test_input_embedding_kernel<int8_t>();
        } else {
            test_input_embedding_kernel<float>();
        }
    } else {
        test_input_embedding_kernel<float>();
    }
    
    return 0;
}
