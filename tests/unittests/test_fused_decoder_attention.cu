#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include "src/kernels/fused_decoder_self_attention.h"
#include "src/utils/macro.h"

// bug1: MUST add CHECK to cudaMemcpy to see if its work well
// this CPU implementation still exists bugs.
// when you are implementing LLMs inference on CPU, you can reuse the CPU kernel and test its correctness
// Usage:
// `./test_fused_decoder_attention` to test fp32 kernel
// `./test_fused_decoder_attention fp16` to test fp16 kernel
// `./test_fused_decoder_attention int8` to test int8 kernel

template <typename T>
void CPUMaskedAttn(T *q,
                   T *k,
                   T *v,
                   T *k_cache,
                   T *v_cache,
                   float *mha_output,
                   const int batch_size,
                   const int num_heads,
                   const int head_size,
                   int step)
{
    int batch_stride = num_heads * head_size;
    int head_stride = head_size;
    int cache_offset = batch_size * batch_stride;
    int block_nums = batch_size * num_heads;
    float scale = rsqrt(float(head_size));

    const T *q_mem = q;
    const T *k_mem = k;
    const T *v_mem = v;

    // tmp buffer
    float *sqk = (float *)malloc(sizeof(float) * (block_nums * (3 * head_size + step)));
    float *sq = sqk;
    float *sk = sq + block_nums * head_size;
    float *logits = sk + block_nums * head_size;
    float *sv = logits + block_nums * step;
    for (int batch_id = 0; batch_id < batch_size; batch_id++)
    {
        for (int head_id = 0; head_id < num_heads; head_id++)
        {
            float row_max = 0.0f;
            for (int iter = 0; iter < step; iter++)
            {
                float attn_score = 0.0f;
                for (int tid = 0; tid < head_size; tid++)
                {
                    int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;
                    sk[qkv_offset] = (float)k_cache[iter * cache_offset + qkv_offset];
                    if (iter == step - 1)
                    {
                        k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
                        sk[qkv_offset] = (float)k_mem[qkv_offset];
                    }

                    sq[qkv_offset] = (float)q_mem[qkv_offset];
                    float qk = sq[qkv_offset] * sk[qkv_offset] * scale;
                    attn_score += qk;
                }
                logits[batch_id * num_heads * step + head_id * step + iter] = attn_score;
                row_max = std::max(attn_score, row_max);
            }
            float fenzi = 0.0f;
            float fenmu = 0.0f;
            for (int iter = 0; iter < step; iter++)
            { 
                fenzi = expf(logits[batch_id * num_heads * step + head_id * step + iter] - row_max);
                fenmu += fenzi;
            }
            for (int iter = 0; iter < step; iter++)
            { 
                logits[batch_id * num_heads * step + head_id * step + iter] = (float)(fenzi / fenmu);
            }
            
            for (int tid = 0; tid < head_size; tid++)
            {
                float O = 0.0f;
                int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;
                for (int iter = 0; iter < step; iter++)
                {
                    sv[qkv_offset] = (float)v_cache[iter * cache_offset + qkv_offset];
                    if (iter == step - 1)
                    {
                        v_cache[iter * cache_offset + qkv_offset] = v_mem[qkv_offset];
                        sv[qkv_offset] = (float)v_mem[qkv_offset];
                    }
                    O += sv[qkv_offset] * logits[batch_id * num_heads * step + head_id * step + iter];
                }
                mha_output[qkv_offset] = O;
            }
        }
    }

    free(sqk);
}

template <typename T>
bool CheckResult(float *CPUoutput, T *GPUoutput, int output_size)
{
    float tolerance = std::is_same<T, int8_t>::value ? 1e-1f : 
                      (std::is_same<T, half>::value ? 1e-2f : 1e-6f);
    for (int i = 0; i < output_size; i++)
    {
        float GPUres = (float)GPUoutput[i];
        if (fabs(CPUoutput[i] - GPUres) > tolerance)
        {
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUres);
            return false;
        }
    }
    return true;
}

template<typename T>
void test_fused_attention_kernel() {
    constexpr int batch_size = 1;
    constexpr int head_size = 4;
    constexpr int num_heads = 2;
    constexpr int kv_num_heads = 2;
    constexpr int max_seq_len = 4;
    int h_step = 4;
    int h_layer_id = 0;
    int rotary_embedding_dim = 128;
    float rotary_embedding_base = 10000;
    int max_position_embeddings = 2048;
    bool use_dynamic_ntk = false;
    
    printf("Testing %s fused decoder attention kernel\n", typeid(T).name());
    
    T *h_qkv;                                                                                                                                 
    T *d_qkv;                                                                                                                                 
    int qkv_size = batch_size * (2 * kv_num_heads + num_heads) * head_size;                                                                       
    h_qkv = (T *)malloc(sizeof(T) * qkv_size);                                                                                            
    cudaMalloc((void **)&d_qkv, sizeof(T) * qkv_size);                                                                                        
    T *h_kcache;                                                                                                                              
    T *d_kcache;                                                                                                                              
    int kcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                                        
    h_kcache = (T *)malloc(sizeof(T) * kcache_size);                                                                                      
    cudaMalloc((void **)&d_kcache, sizeof(T) * kcache_size);                                                                                  
    T *h_vcache;                                                                                                                              
    T *d_vcache;                                                                                                                              
    int vcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                                        
    h_vcache = (T *)malloc(sizeof(T) * vcache_size);                                                                                      
    cudaMalloc((void **)&d_vcache, sizeof(T) * vcache_size);                                                                                  
    
    for (int i = 0; i < qkv_size; i++)                                                                                                            
    {                                                                                                                                             
        if (i < batch_size * num_heads * head_size)                                                                                               
        {                                                                                                                                         
            if (i < batch_size * num_heads * head_size / 2)                                                                                       
            {                                                                                                                                     
                h_qkv[i] = (T)(i + 1);                                                                                                        
            }                                                                                                                                     
            else                                                                                                                                  
            {                                                                                                                                     
                h_qkv[i] = (T)(i - 3) / (T)10;                                                                                                   
            }                                                                                                                                     
        }                                                                                                                                         
        else if (i < batch_size * (num_heads + kv_num_heads) * head_size)                                                                         
        {                                                                                                                                         
            if (i < batch_size * (num_heads + kv_num_heads / 2) * head_size)                                                                      
            {                                                                                                                                     
                h_qkv[i] = (T)(i + 5);                                                                                                        
            }                                                                                                                                     
            else                                                                                                                                  
            {                                                                                                                                     
                h_qkv[i] = (T)(i + 1) / (T)10;                                                                                                   
            }                                                                                                                                     
        }                                                                                                                                         
        else if (i < batch_size * (num_heads + kv_num_heads * 2) * head_size)                                                                     
        {                                                                                                                                         
            if (i < batch_size * (num_heads + kv_num_heads + kv_num_heads / 2) * head_size)                                                       
            {                                                                                                                                     
                h_qkv[i] = (T)(i - 3);                                                                                                        
            }                                                                                                                                     
            else                                                                                                                                  
            {                                                                                                                                     
                h_qkv[i] = (T)(i - 7) / (T)10;                                                                                                   
            }                                                                                                                                     
        }                                                                                                                                         
    }                                                                                                                                             
    T *h_q = h_qkv;                                                                                                                           
    T *h_k = h_q + batch_size * num_heads * head_size;                                                                                        
    T *h_v = h_k + batch_size * (kv_num_heads + num_heads) * head_size;                                                                       
    for (int i = 0; i < (kcache_size * h_step) / max_seq_len; i++)                                                                                
    {                                                                                                                                             
        if (i < kcache_size / 2)                                                                                                                  
        {                                                                                                                                         
            h_kcache[i] = (T)(i + 1);                                                                                                         
            h_vcache[i] = (T)(i + 1);                                                                                                         
        }                                                                                                                                        
        else                                                                                                                                     
        {                                                                                                                                        
            h_kcache[i] = (T)(i - kcache_size / 2 + 1) / (T)10;                                                                                 
            h_vcache[i] = (T)(i - kcache_size / 2 + 1) / (T)10;                                                                                 
        }                                                                                                                                        
    }                                                                                                                                            
    T *h_o;                                                                                                                                  
    T *d_o;                                                                                                                                  
    int o_size = batch_size * num_heads * head_size;                                                                                             
    h_o = (T *)malloc(sizeof(T) * o_size);                                                                                               
    cudaMalloc((void **)&d_o, sizeof(T) * o_size);                                                                                           
    bool *h_finished = (bool *)malloc(sizeof(bool) * batch_size);                                                                                
    bool *d_finished;                                                                                                                            
    cudaMalloc((void **)&d_finished, sizeof(bool) * batch_size);                                                                                 
    for (int i = 0; i < batch_size; i++)                                                                                                         
    {                                                                                                                                            
        h_finished[i] = static_cast<bool>(0);                                                                                                    
    }                                                                                                                                            
    T *h_qkv_bias = (T *)malloc(sizeof(T) * (2 * kv_num_heads + num_heads) * head_size);                                             
    T *d_qkv_bias;                                                                                                                           
    cudaMalloc((void **)&d_qkv_bias, sizeof(T) * (2 * kv_num_heads + num_heads) * head_size);                                                
    for (int i = 0; i < (2 * kv_num_heads + num_heads) * head_size; i++)                                                                         
    {                                                                                                                                            
        h_qkv_bias[i] = (T)0.0f;                                                                                                             
    }                                                                                                                                            
    cudaMemcpy(d_qkv, h_qkv, sizeof(T) * batch_size * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice);                   
    cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(T) * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice);                      
    cudaMemcpy(d_finished, h_finished, sizeof(bool) * batch_size, cudaMemcpyHostToDevice);                                                       
    cudaMemcpy(d_kcache, h_kcache, sizeof(T) * kcache_size, cudaMemcpyHostToDevice);                                                         
    cudaMemcpy(d_vcache, h_vcache, sizeof(T) * vcache_size, cudaMemcpyHostToDevice);                                                         
    DataType type = getTensorType<T>();                                                                                                      
    DataType type_bool = getTensorType<bool>();                                                                                                  
    DataType type_int = getTensorType<int>();                                                                                                    
    TensorWrapper<T> *qkv = new TensorWrapper<T>(GPU, type, {batch_size, num_heads + 2 * kv_num_heads, head_size}, d_qkv);               
    TensorWrapper<T> *kcache = new TensorWrapper<T>(GPU, type, {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, d_kcache);
    TensorWrapper<T> *vcache = new TensorWrapper<T>(GPU, type, {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, d_vcache);
    TensorWrapper<bool> *finished = new TensorWrapper<bool>(GPU, type_bool, {batch_size}, d_finished);                                           
    TensorWrapper<int> *step = new TensorWrapper<int>(CPU, type_int, {1}, &h_step);                                                              
    TensorWrapper<int> *layer_id = new TensorWrapper<int>(CPU, type_int, {1}, &h_layer_id);                                                      
    TensorWrapper<T> *mha_output = new TensorWrapper<T>(GPU, type, {batch_size, num_heads, head_size}, d_o);                             
    BaseWeight<T> qkv_weight;                                                                                                                
    qkv_weight.bias = d_qkv_bias;                                                                                                                
    LLaMAAttentionStaticParams params;                                                                                                           
    params.rotary_embedding_dim = rotary_embedding_dim;                                                                                          
    params.rotary_embedding_base = rotary_embedding_base;                                                                                        
    params.max_position_embeddings = max_position_embeddings;                                                                                    
    params.use_dynamic_ntk = false;                                                                                                              
    launchDecoderMaskedMHA(qkv, qkv_weight, layer_id, kcache, vcache, finished, step, mha_output, params);                                       
    CHECK(cudaMemcpy(h_o, d_o, sizeof(T) * o_size, cudaMemcpyDeviceToHost));                                                                 
    float *CPU_output = (float *)malloc(sizeof(float) * o_size);                                                                                 
    CPUMaskedAttn<T>(h_q, h_k, h_v, h_kcache, h_vcache, CPU_output, batch_size, num_heads, head_size, h_step);                               
    bool is_true = CheckResult<T>(CPU_output, h_o, o_size);                                                                                  
    if (is_true)                                                                                                                                 
    {                                                                                                                                            
        printf("test passed\n");                                                                                                                   
    }                                                                                                                                            
    else                                                                                                                                         
    {                                                                                                                                            
        printf("test failed\n");                                                                                                                   
    }                                                                                                                                            
    free(h_qkv);                                                                                                                                 
    free(h_kcache);                                                                                                                              
    free(h_vcache);                                                                                                                              
    free(h_o);                                                                                                                                   
    free(CPU_output);                                                                                                                            
    free(h_finished);                                                                                                                            
    free(h_qkv_bias);                                                                                                                            
    cudaFree(d_finished);                                                                                                                        
    cudaFree(d_qkv);                                                                                                                             
    cudaFree(d_o);                                                                                                                               
    cudaFree(d_kcache);                                                                                                                          
    cudaFree(d_vcache);
    cudaFree(d_qkv_bias);
    
    delete qkv;
    delete kcache;
    delete vcache;
    delete finished;
    delete step;
    delete layer_id;
    delete mha_output;
}

int main(int argc, char *argv[])
{
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "fp16") {
            test_fused_attention_kernel<half>();
        } else if (arg == "int8") {
            test_fused_attention_kernel<int8_t>();
        } else {
            test_fused_attention_kernel<float>();
        }
    } else {
        test_fused_attention_kernel<float>();
    }
    
    return 0;
}
