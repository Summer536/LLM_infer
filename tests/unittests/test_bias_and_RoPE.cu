#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <cstring>     // strcmp

#include "src/kernels/qkv_bias_and_RoPE.h"
#include "src/weights/llama/attention_weights.h"
#include "src/utils/macro.h"
// Not sure CPU implementation is absolutely right and the GPU kernel is right compared with HF.

// Helper function for rounding float to int (CPU version)
inline int cpu_float2int_rn(float val) {
    return static_cast<int>(roundf(val));
}

template<typename T>
void initializeBiasRoPEData(T* data, int size, float scale = 1.0f) {
    for(int i = 0; i < size; i++) {
        if constexpr (std::is_same_v<T, int8_t>) {
            // For INT8, use quantized values
            float val = (float)(i % 100 + 1) * scale * 0.01f; // Small values for QKV
            data[i] = static_cast<T>(cpu_float2int_rn(val * INT8_SCALE_FACTOR));
        } else if constexpr (std::is_same_v<T, half>) {
            // For FP16
            data[i] = __float2half((float)(i % 100 + 1) * scale * 0.01f);
        } else {
            // For FP32  
            data[i] = static_cast<T>((i % 100 + 1)) * scale * 0.01f;
        }
    }
}

// CPU implementation for verification - always in float precision
// This should match the GPU kernel logic exactly
void CPUfunc(float* q,
                float* k,
                float* v,
                float* QKV,
                const float* qkv_bias,
                const int*   padding_offset,
                const int*   history_length,
                const int*   input_length,
                const int    batch_size,
                const int    seq_len,
                const int    token_num,
                const int    head_num,
                const int    kv_head_num,
                const int    head_size,
                const int    rotary_embedding_dim,
                float        rotary_embedding_base) {
    
    // Follow the same logic as GPU kernel
    int qkv_head_num = head_num + 2 * kv_head_num;
    
    for (int token_id = 0; token_id < token_num; token_id++) {
        int token_padding_offset = padding_offset[token_id];
        int dst_token_id = token_id + token_padding_offset;
        int batch_id = dst_token_id / seq_len;
        int local_token_id = dst_token_id % seq_len;
        
        const int cur_seq_history_len = history_length[batch_id];
        const int timestep = cur_seq_history_len + local_token_id;
        
        for (int head_id = 0; head_id < head_num; head_id++) {
            for (int tid = 0; tid < head_size; tid++) {
                // Calculate indices same as GPU kernel
                int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid;
                int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size;
                int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size + kv_head_num * head_size;
                
                int dst_q_id = batch_id * seq_len * head_num * head_size +
                               head_id * seq_len * head_size +
                               local_token_id * head_size + tid;
                               
                int dst_kv_id = batch_id * seq_len * kv_head_num * head_size +
                                head_id * seq_len * head_size +
                                local_token_id * head_size + tid;
                
                // V value (no RoPE needed)
                if (head_id < kv_head_num) {
                    v[dst_kv_id] = QKV[v_id];
                }
                
                // RoPE processing
                if (tid < rotary_embedding_dim / 2) {
                    // Calculate RoPE frequencies
                    float inv_freq = timestep / powf(rotary_embedding_base, (tid * 2) / (float)rotary_embedding_dim);
                    float cos_val = cos(inv_freq);
                    float sin_val = sin(inv_freq);
                    
                    // Q RoPE
                    float q_val = QKV[q_id];
                    float q_rotate_val = QKV[q_id + head_size / 2];
                    float q_rotated_x = cos_val * q_val - sin_val * q_rotate_val;
                    float q_rotated_y = cos_val * q_rotate_val + sin_val * q_val;
                    
                    q[dst_q_id] = q_rotated_x;
                    q[dst_q_id + head_size / 2] = q_rotated_y;
                    
                    // K RoPE  
                    if (head_id < kv_head_num) {
                        float k_val = QKV[k_id];
                        float k_rotate_val = QKV[k_id + head_size / 2];
                        float k_rotated_x = cos_val * k_val - sin_val * k_rotate_val;
                        float k_rotated_y = cos_val * k_rotate_val + sin_val * k_val;
                        
                        k[dst_kv_id] = k_rotated_x;
                        k[dst_kv_id + head_size / 2] = k_rotated_y;
                    }
                } else {
                    // Non-RoPE dimensions, direct copy
                    q[dst_q_id] = QKV[q_id];
                    if (head_id < kv_head_num) {
                        k[dst_kv_id] = QKV[k_id];
                    }
                }
            }
        }
    }
}

template<typename T>
bool CheckResult(float* q_cpu, float* k_cpu, T* q_gpu, T* k_gpu, 
                const int q_size, const int k_size) {
    int errors = 0;
    constexpr int MAX_ERRORS_TO_PRINT = 10;
    
    // Set tolerance based on data type
    float tolerance;
    if constexpr (std::is_same_v<T, int8_t>) {
        tolerance = 0.2f; // Higher tolerance for INT8 due to quantization in RoPE
    } else if constexpr (std::is_same_v<T, half>) {
        tolerance = 5e-2f; // Medium tolerance for FP16
    } else {
        tolerance = 1e-5f; // High precision for FP32
    }
    
    printf("===== Checking results with tolerance: %f =====\n", tolerance);
    
    // Check Q results
    for(int i = 0; i < q_size; i++) {
        float gpu_val_f;
        if constexpr (std::is_same_v<T, int8_t>) {
            gpu_val_f = static_cast<float>(q_gpu[i]) * INT8_INV_SCALE_FACTOR;
        } else if constexpr (std::is_same_v<T, half>) {
            gpu_val_f = __half2float(q_gpu[i]);
        } else {
            gpu_val_f = static_cast<float>(q_gpu[i]);
        }
        
        if(fabs(q_cpu[i] - gpu_val_f) > tolerance){
            if (errors < MAX_ERRORS_TO_PRINT) {
                printf("Q error at %d: CPU = %f, GPU = %f, diff = %f\n", 
                        i, q_cpu[i], gpu_val_f, fabs(q_cpu[i] - gpu_val_f));
            }
            errors++;
        }
    }
    
    // Check K results
    for(int i = 0; i < k_size; i++) {
        float gpu_val_f;
        if constexpr (std::is_same_v<T, int8_t>) {
            gpu_val_f = static_cast<float>(k_gpu[i]) * INT8_INV_SCALE_FACTOR;
        } else if constexpr (std::is_same_v<T, half>) {
            gpu_val_f = __half2float(k_gpu[i]);
        } else {
            gpu_val_f = static_cast<float>(k_gpu[i]);
        }
        
        if(fabs(k_cpu[i] - gpu_val_f) > tolerance){
            if (errors < MAX_ERRORS_TO_PRINT) {
                printf("K error at %d: CPU = %f, GPU = %f, diff = %f\n", 
                        i, k_cpu[i], gpu_val_f, fabs(k_cpu[i] - gpu_val_f));
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("Found %d errors (diff > %f)\n", errors, tolerance);
        return false;
    }
    return true;
}

template<typename T>
void testBiasAndRoPE(const std::string& type_name) {
    std::cout << "\n=== Testing QKV Bias and RoPE with " << type_name << " ===\n" << std::endl;
    
    const int batch_size = 1;
    const int seq_len = 8; // Reduced for easier testing  
    const int token_num = batch_size * seq_len;
    const int head_num = 8; // Reduced for easier testing
    const int kv_head_num = 8;
    const int head_size = 64; // Reduced for easier testing
    const int rotary_embedding_dim = 64;
    const int rotary_embedding_base = 10000;
    const int max_position_embeddings = 2048;
    
    // Calculate sizes
    const int q_size = batch_size * seq_len * head_num * head_size;
    const int k_size = batch_size * seq_len * kv_head_num * head_size;
    const int v_size = batch_size * seq_len * kv_head_num * head_size;
    const int qkv_size = token_num * (head_num + 2 * kv_head_num) * head_size;
    
    printf("Configuration: batch_size=%d, seq_len=%d, head_num=%d, kv_head_num=%d, head_size=%d\n",
           batch_size, seq_len, head_num, kv_head_num, head_size);
    printf("Sizes: q_size=%d, k_size=%d, v_size=%d, qkv_size=%d\n", q_size, k_size, v_size, qkv_size);
    
    // Allocate host memory
    int* padding_offset = (int*)malloc(sizeof(int) * token_num);
    int* history_length = (int*)malloc(sizeof(int) * batch_size);
    int* input_length = (int*)malloc(sizeof(int) * batch_size);
    
    T* h_q = (T*)malloc(sizeof(T) * q_size);
    T* h_k = (T*)malloc(sizeof(T) * k_size);
    T* h_v = (T*)malloc(sizeof(T) * v_size);
    T* h_QKV = (T*)malloc(sizeof(T) * qkv_size);
    T* h_qkv_bias = (T*)malloc(sizeof(T) * (head_num + 2 * kv_head_num) * head_size);
    
    // For CPU verification
    float* h_q_cpu = (float*)malloc(sizeof(float) * q_size);
    float* h_k_cpu = (float*)malloc(sizeof(float) * k_size);
    float* h_v_cpu = (float*)malloc(sizeof(float) * v_size);
    float* h_QKV_cpu = (float*)malloc(sizeof(float) * qkv_size);
    float* h_qkv_bias_cpu = (float*)malloc(sizeof(float) * (head_num + 2 * kv_head_num) * head_size);
    
    // Initialize data
    initializeBiasRoPEData(h_QKV, qkv_size);
    initializeBiasRoPEData(h_qkv_bias, (head_num + 2 * kv_head_num) * head_size);
    
    // Convert to float for CPU computation
    for(int i = 0; i < qkv_size; i++){
        if constexpr (std::is_same_v<T, int8_t>) {
            h_QKV_cpu[i] = static_cast<float>(h_QKV[i]) * INT8_INV_SCALE_FACTOR;
        } else if constexpr (std::is_same_v<T, half>) {
            h_QKV_cpu[i] = __half2float(h_QKV[i]);
        } else {
            h_QKV_cpu[i] = static_cast<float>(h_QKV[i]);
        }
    }
    
    for(int i = 0; i < (head_num + 2 * kv_head_num) * head_size; i++){
        if constexpr (std::is_same_v<T, int8_t>) {
            h_qkv_bias_cpu[i] = static_cast<float>(h_qkv_bias[i]) * INT8_INV_SCALE_FACTOR;
        } else if constexpr (std::is_same_v<T, half>) {
            h_qkv_bias_cpu[i] = __half2float(h_qkv_bias[i]);
        } else {
            h_qkv_bias_cpu[i] = static_cast<float>(h_qkv_bias[i]);
        }
    }
    
    for(int i = 0; i < batch_size; i++){
        input_length[i] = seq_len;
        history_length[i] = 0;
    }
    for(int i = 0; i < token_num; i++){
        padding_offset[i] = 0;
    }

    // Allocate device memory
    int* d_padding_offset;
    int* d_history_length; 
    int* d_input_length;
    T* d_q;
    T* d_k;
    T* d_v;
    T* d_QKV;
    T* d_qkv_bias;
    
    CHECK(cudaMalloc((void**)&d_padding_offset, sizeof(int) * token_num));
    CHECK(cudaMalloc((void**)&d_history_length, sizeof(int) * batch_size));
    CHECK(cudaMalloc((void**)&d_input_length, sizeof(int) * batch_size));
    CHECK(cudaMalloc((void**)&d_q, sizeof(T) * q_size));
    CHECK(cudaMalloc((void**)&d_k, sizeof(T) * k_size));
    CHECK(cudaMalloc((void**)&d_v, sizeof(T) * v_size));
    CHECK(cudaMalloc((void**)&d_QKV, sizeof(T) * qkv_size));
    CHECK(cudaMalloc((void**)&d_qkv_bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size));

    // Copy to device
    CHECK(cudaMemcpy(d_input_length, input_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_history_length, history_length, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_padding_offset, padding_offset, sizeof(int) * token_num, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_QKV, h_QKV, sizeof(T) * qkv_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(T) * (head_num + 2 * kv_head_num) * head_size, cudaMemcpyHostToDevice));
    
    // Create tensor wrappers
    DataType type = getTensorType<T>(); 
    DataType type_int = getTensorType<int>(); 
    
    TensorWrapper<T>* q_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, seq_len, head_size}, d_q);
    TensorWrapper<T>* k_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, seq_len, head_size}, d_k);
    TensorWrapper<T>* v_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, seq_len, head_size}, d_v);
    TensorWrapper<T>* QKV_buf = new TensorWrapper<T>(Device::GPU, type, {token_num, head_num + 2 * kv_head_num, head_size}, d_QKV);
    
    LLaMAattentionWeights<T> attn_weights;
    attn_weights.qkv.bias = d_qkv_bias;
    
    TensorWrapper<int>* input_length_buf = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_input_length);
    TensorWrapper<int>* history_length_buf = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_history_length);
    TensorWrapper<int>* padding_offset_buf = new TensorWrapper<int>(Device::GPU, type_int, {token_num}, d_padding_offset);
    
    LLaMAAttentionStaticParams params;
    params.rotary_embedding_dim = rotary_embedding_dim;
    params.rotary_embedding_base = rotary_embedding_base;
    params.max_position_embeddings = max_position_embeddings;
    params.use_dynamic_ntk = false;
    
    // Launch GPU kernel
    std::cout << "Launching " << type_name << " GPU kernel..." << std::endl;
    launchAddFusedQKVBiasTransposeAndRoPE(q_buf,
                                          k_buf,
                                          v_buf,
                                          QKV_buf,
                                          attn_weights.qkv,
                                          padding_offset_buf,
                                          history_length_buf,
                                          input_length_buf,
                                          params);
    CHECK(cudaDeviceSynchronize());
    std::cout << "GPU kernel completed" << std::endl;
    
    // Copy results back
    CHECK(cudaMemcpy(h_q, d_q, sizeof(T) * q_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_k, d_k, sizeof(T) * k_size, cudaMemcpyDeviceToHost));
    
    // Run CPU reference
    std::cout << "Running CPU reference..." << std::endl;
    CPUfunc(h_q_cpu,
            h_k_cpu,
            h_v_cpu,
            h_QKV_cpu,
            h_qkv_bias_cpu,
            padding_offset,
            history_length,
            input_length,
            batch_size,
            seq_len,
            token_num,
            head_num,
            kv_head_num,
            head_size,
            rotary_embedding_dim,
            rotary_embedding_base);
    std::cout << "CPU reference completed" << std::endl;
    
    // Check results
    bool is_right = CheckResult<T>(h_q_cpu, h_k_cpu, h_q, h_k, q_size, k_size);
    std::cout << type_name << " QKV Bias and RoPE test " << (is_right ? "PASSED" : "FAILED") << std::endl;
    
    // Cleanup
    free(padding_offset);
    free(history_length);
    free(input_length);
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_QKV);
    free(h_qkv_bias);
    free(h_q_cpu);
    free(h_k_cpu);
    free(h_v_cpu);
    free(h_QKV_cpu);
    free(h_qkv_bias_cpu);
    
    delete q_buf;
    delete k_buf;
    delete v_buf;
    delete QKV_buf;
    delete input_length_buf;
    delete history_length_buf;
    delete padding_offset_buf;
    
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_QKV);
    cudaFree(d_qkv_bias);
    cudaFree(d_padding_offset);
    cudaFree(d_history_length);
    cudaFree(d_input_length);
    
    std::cout << type_name << " test completed!" << std::endl;
}

// Usage: `./test_bias_and_RoPE [fp32|fp16|int8]` to test different precisions
int main(int argc, char* argv[]) {
    std::string data_type = "fp32";
    if (argc > 1) {
        data_type = argv[1];
    }
    
    std::cout << "Testing QKV Bias and RoPE with data type: " << data_type << std::endl;
    
    if (data_type == "fp32") {
        testBiasAndRoPE<float>("FP32");
    } else if (data_type == "fp16") {
        testBiasAndRoPE<half>("FP16");
    } else if (data_type == "int8") {
        testBiasAndRoPE<int8_t>("INT8");
    } else {
        std::cerr << "Unknown data type: " << data_type << std::endl;
        std::cerr << "Usage: " << argv[0] << " [fp32|fp16|int8]" << std::endl;
        return 1;
    }
    
    return 0;
}
