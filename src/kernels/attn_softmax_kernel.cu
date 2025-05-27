#include "src/kernels/attn_softmax_kernel.h"
#include "src/utils/cuda_debug_utils.cuh"
#include <float.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
// attention_score,    (batch_size, head_num, q_length, k_length), softmax output.  //Output
// qk,                 (batch_size, head_num, q_length, k_length), QK^T.            //Input
// attention_mask,     (batch_size, q_length, k_length), attention mask.            //Input
template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
};

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T warpReduce(T val)
{
    for (int mask = 32 / 2; mask > 0; mask /= 2)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T blockReduce(T val)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31) / 32;
    static __shared__ T warp[64];
    val = warpReduce<ReductionOp, T>(val);
    if (lane_id == 0)
    {
        warp[warp_id] = val;
    }
    __syncthreads();
    float warp_val = tid < warp_nums ? warp[tid] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}

template <typename T, int NUMS_PER_THREAD_PER_ROW>  
__global__ void ScaleMaskAndSoftmax_float(T *attn_score,
                                          T *qk,
                                          T *mask,
                                          int batch_size,
                                          int head_nums,
                                          int q_len,
                                          int k_len,
                                          float scale)
{
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;

    if (threadIdx.x >= k_len)  
    {
        return;
    }
    
    __shared__ float inv_sum, s_max;  

    for (int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x) 
    {
        int qk_offset = 0;
        int mask_offset = 0;
        T qk_data = static_cast<T>(0);
        T mask_data = static_cast<T>(0);
        T thread_max = FLT_MIN;
        T data[NUMS_PER_THREAD_PER_ROW]; 
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++) 
        {   
            ///////////////////////////////////////////////////////attention mask///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //(batch_size, head_num, q_length, k_length), QK^T.
            qk_offset = batch_id * head_nums * q_len * k_len + head_id * q_len * k_len + row_start * k_len + col_start * blockDim.x + threadIdx.x;

            qk_data = qk[qk_offset];
            
            //(batch_size, q_length, k_length), attention mask.    
            mask_offset = batch_id * q_len * k_len + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            mask_data = mask[mask_offset];
            ///////////////////////////////////////////////////////attention mask///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            
            ///////////////////////////////////////////////////////  scale  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // https://www.zhihu.com/question/472323371/answer/2001223766
            data[col_start] = scale * qk_data + (1 - mask_data) * (-10000.0f); 
            ///////////////////////////////////////////////////////  scale  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            thread_max = fmax(data[col_start], thread_max); 
        }

        T max_val = blockReduce<MaxOp, T>(thread_max);
        if (threadIdx.x == 0) 
        {
            s_max = max_val;
            // debug info,printf("row max = %f\n", s_max);
        }
        __syncthreads();  

        T thread_sum = 0.0f;
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++) 
        {
            qk_offset = batch_id * head_nums * q_len * k_len + head_id * q_len * k_len + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            mask_offset = batch_id * q_len * k_len + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            data[col_start] = expf(data[col_start] - s_max); 
            thread_sum += data[col_start]; 
        }

        T sum = blockReduce<SumOp, T>(thread_sum);
        if (threadIdx.x == 0)
        {
            inv_sum = 1 / (sum + 1e-6f); 
            // debug info, printf("row sum = %f\n", sum);
        }
        __syncthreads();
        ///////////////////////////////////////////////////////  softmax  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++)
        {   
            //因为attn_score的shape和QK^T的shape是一样的。因此这里我们直接用qk_offset的偏移去访问对应的便宜量attn_score[qk_offset]
            qk_offset = batch_id * head_nums * q_len * k_len + head_id * q_len * k_len + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            attn_score[qk_offset] = (data[col_start] * inv_sum); //data值乘以softmax的分母倒数，得到的最终结果输出到attn_score(融合算子最终输出矩阵)中
        }
        ///////////////////////////////////////////////////////  softmax  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
}


///////////////////////////////////////////////////////////////////以下是FP16版本！////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////      与FP32不同的部分将用会///////来标注            //////////////////////////////
// cant partial specialize in func
template <typename T_half, int NUMS_PER_THREAD_PER_ROW>
__global__ void ScaleMaskAndSoftmax_half(T_half *attn_score,
                                         T_half *qk,
                                         T_half *mask,
                                         int batch_size,
                                         int head_nums,
                                         int q_len,
                                         int k_len,
                                         float scale)
{
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    // note: NUMS_PER_THREAD_PER_ROW must be a constant value that known at compile time, following expr is invalid
    // const int NUMS_PER_THREAD_PER_ROW = ceil(k_len / blockDim.x);
    int vec_size = Vec<T_half>::size;          ///////FP16是以向量长度为2的力度来读写并且计算的。因此我们需要拿到向量类型和大小
    using Vec_t = typename Vec<T_half>::Type;  ///////这个过程和RMSNorm中用到向量的时候类似(位于/src/kernels/rmsnorm_kernel.cu)

    ///////
    Vec_t* attn_score_vec = reinterpret_cast<Vec_t*>(attn_score);
    Vec_t* qk_buf_vec = reinterpret_cast<Vec_t*>(qk);
    Vec_t* attn_mask_vec  = reinterpret_cast<Vec_t*>(mask);

    ///////scalar_cast_vec:将常量转换为两个或四个向量的方法，把1、-inf、scale都做了一个向量化，方便half2的计算
    Vec_t ONE = scalar_cast_vec<Vec_t>(__float2half(1.0f));
    Vec_t NEG_INF = scalar_cast_vec<Vec_t>(__float2half(-10000.0f));
    Vec_t scale_vec = scalar_cast_vec<Vec_t>(__float2half(scale));

    __shared__ float inv_sum, s_max;
    // warning: remember 1st priority thing is filtering the out-of-boundary threads
    if (threadIdx.x * vec_size >= k_len)
    {
        return;
    }

    for (int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x) 
    {
        int qk_offset = 0;
        int mask_offset = 0;
        Vec_t qk_data;
        Vec_t mask_data;
        float thread_max = FLT_MIN;
        Vec_t data[NUMS_PER_THREAD_PER_ROW]; 
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++) 
        {
            qk_offset = batch_id * head_nums * q_len * k_len / 2 + head_id * q_len * k_len / 2  + row_start * k_len / 2 + col_start * blockDim.x + threadIdx.x;
            qk_data = qk_buf_vec[qk_offset];

            mask_offset = batch_id * q_len * k_len / 2 + row_start * k_len / 2 + col_start * blockDim.x + threadIdx.x;
            mask_data = attn_mask_vec[mask_offset];
            Vec_t mask_vec_reg= __hmul2(__hsub2(ONE, mask_data), NEG_INF);

            data[col_start] = __hadd2(__hmul2(scale_vec, qk_data), mask_vec_reg);
            thread_max = fmax(fmax((float)data[col_start].x, (float)data[col_start].y), thread_max);
        }
        // warp/block reduce
        float max_val = blockReduce<MaxOp, float>(thread_max);     //////这里为什么取最大值和下方取求和的时候采用的是float(FP32)而不用FP16了呢？  主要是怕数据溢出，用FP32保险点
        if (threadIdx.x == 0)
        {
            s_max = max_val;
            //printf("row max = %f\n", s_max);
        }
        __syncthreads();
        // thread local fenzi/fenmu
        float thread_sum = 0.0f;
        // for(int col_start = threadIdx.x; col_start < k_len; col_start += blockDim.x){
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++)
        {
            // debug info to see useless threads if its available,printf("blockIdx.x=%d, threadIdx.x=%d\n",blockIdx.x, threadIdx.x);
            data[col_start] = h2exp(__hsub2(data[col_start], scalar_cast_vec<Vec_t>(s_max)));      /////// half2的exp函数      scalar_cast_vec：转标量为向量类型
            thread_sum += (float)(__hadd(data[col_start].x, data[col_start].y));                   /////// half2先局部做一个加法，然后再累加
            // debug info,printf("after, data[%d]=%f, thread_sum = %f\n",col_start, data[col_start], thread_sum);
        }
        // row sum
        float sum = blockReduce<SumOp, float>(thread_sum);
        if (threadIdx.x == 0)
        {
            inv_sum = 1 / (sum + 1e-6f); // sum(fenmu) need to add a small value to avoid NAN
            //printf("row sum = %f\n", sum);
        }
        __syncthreads();
        // write back into gmem
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++)
        {
            qk_offset = batch_id * head_nums * q_len * k_len / 2 + head_id * q_len * k_len / 2 + row_start * k_len / 2 + col_start * blockDim.x + threadIdx.x;
            attn_score_vec[qk_offset] = __hmul2(data[col_start], scalar_cast_vec<Vec_t>(inv_sum));  
        }
    }
}


///////////////////////////////////////////////////////////////////以下是INT8版本！////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////      与FP32/FP16不同的部分将用会///////来标注            //////////////////////////////
template <typename T_int8, int NUMS_PER_THREAD_PER_ROW>
__global__ void ScaleMaskAndSoftmax_int8(T_int8 *attn_score,
                                         T_int8 *qk,
                                         T_int8 *mask,
                                         int batch_size,
                                         int head_nums,
                                         int q_len,
                                         int k_len,
                                         float scale)
{
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    
    int vec_size = Vec<T_int8>::size;          ///////INT8是以向量长度为4的粒度来读写并且计算的
    using Vec_t = typename Vec<T_int8>::Type;  ///////这个过程和RMSNorm中用到向量的时候类似

    ///////
    Vec_t* attn_score_vec = reinterpret_cast<Vec_t*>(attn_score);
    Vec_t* qk_buf_vec = reinterpret_cast<Vec_t*>(qk);
    Vec_t* attn_mask_vec  = reinterpret_cast<Vec_t*>(mask);

    __shared__ float inv_sum, s_max;
    // warning: remember 1st priority thing is filtering the out-of-boundary threads
    if (threadIdx.x * vec_size >= k_len)
    {
        return;
    }

    for (int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x) 
    {
        int qk_offset = 0;
        int mask_offset = 0;
        Vec_t qk_data;
        Vec_t mask_data;
        float thread_max = FLT_MIN;
        Vec_t data[NUMS_PER_THREAD_PER_ROW]; 
        float float_data[NUMS_PER_THREAD_PER_ROW][4]; // Store dequantized values
        
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++) 
        {
            qk_offset = batch_id * head_nums * q_len * k_len / 4 + head_id * q_len * k_len / 4  + row_start * k_len / 4 + col_start * blockDim.x + threadIdx.x;
            qk_data = qk_buf_vec[qk_offset];

            mask_offset = batch_id * q_len * k_len / 4 + row_start * k_len / 4 + col_start * blockDim.x + threadIdx.x;
            mask_data = attn_mask_vec[mask_offset];
            
            // Dequantize and apply scale and mask
            float_data[col_start][0] = scale * ((float)qk_data.x * INT8_INV_SCALE_FACTOR) + (1.0f - (float)mask_data.x * INT8_INV_SCALE_FACTOR) * (-10000.0f);
            float_data[col_start][1] = scale * ((float)qk_data.y * INT8_INV_SCALE_FACTOR) + (1.0f - (float)mask_data.y * INT8_INV_SCALE_FACTOR) * (-10000.0f);
            float_data[col_start][2] = scale * ((float)qk_data.z * INT8_INV_SCALE_FACTOR) + (1.0f - (float)mask_data.z * INT8_INV_SCALE_FACTOR) * (-10000.0f);
            float_data[col_start][3] = scale * ((float)qk_data.w * INT8_INV_SCALE_FACTOR) + (1.0f - (float)mask_data.w * INT8_INV_SCALE_FACTOR) * (-10000.0f);
            
            thread_max = fmax(fmax(fmax(fmax(float_data[col_start][0], float_data[col_start][1]), 
                                       float_data[col_start][2]), float_data[col_start][3]), thread_max);
        }
        
        // warp/block reduce
        float max_val = blockReduce<MaxOp, float>(thread_max);
        if (threadIdx.x == 0)
        {
            s_max = max_val;
        }
        __syncthreads();
        
        // thread local fenzi/fenmu
        float thread_sum = 0.0f;
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++)
        {
            float_data[col_start][0] = expf(float_data[col_start][0] - s_max);
            float_data[col_start][1] = expf(float_data[col_start][1] - s_max);
            float_data[col_start][2] = expf(float_data[col_start][2] - s_max);
            float_data[col_start][3] = expf(float_data[col_start][3] - s_max);
            
            thread_sum += float_data[col_start][0] + float_data[col_start][1] + float_data[col_start][2] + float_data[col_start][3];
        }
        
        // row sum
        float sum = blockReduce<SumOp, float>(thread_sum);
        if (threadIdx.x == 0)
        {
            inv_sum = 1 / (sum + 1e-6f);
        }
        __syncthreads();
        
        // write back into gmem - quantize the softmax results
        for (int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++)
        {
            qk_offset = batch_id * head_nums * q_len * k_len / 4 + head_id * q_len * k_len / 4 + row_start * k_len / 4 + col_start * blockDim.x + threadIdx.x;
            
            // Apply inverse sum and quantize back to int8
            Vec_t result;
            result.x = (int8_t)__float2int_rn((float_data[col_start][0] * inv_sum) * INT8_SCALE_FACTOR);
            result.y = (int8_t)__float2int_rn((float_data[col_start][1] * inv_sum) * INT8_SCALE_FACTOR);
            result.z = (int8_t)__float2int_rn((float_data[col_start][2] * inv_sum) * INT8_SCALE_FACTOR);
            result.w = (int8_t)__float2int_rn((float_data[col_start][3] * inv_sum) * INT8_SCALE_FACTOR);
            
            attn_score_vec[qk_offset] = result;
        }
    }
}


#define LAUNCH_SOFTMAX(dtype, vec_size)                                                                         \
    if (block.x > 2048 && block.x <= 4096)                                                                      \
    {                                                                                                           \
        constexpr int NUMS_PER_THREAD_PER_ROW = 4;                                                              \
        block.x /= 4 * vec_size;                                                                                \
        block.x = (block.x + 32 - 1) / 32 * 32;                                                                 \
        assert(block.x < 1024);                                                                                  \
        ScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data, \
                                                                                     (dtype *)qk->data,         \
                                                                                     (dtype *)mask->data,       \
                                                                                     batch_size,                \
                                                                                     head_nums,                 \
                                                                                     q_length,                  \
                                                                                     k_length,                  \
                                                                                     scale);                    \
    }                                                                                                           \
    else if (block.x > 1024)                                                                                   \
    {                                                                                                           \
        constexpr int NUMS_PER_THREAD_PER_ROW = 2;                                                              \
        block.x /= 2 * vec_size;                                                                                \
        block.x = (block.x + 32 - 1) / 32 * 32;                                                                 \
        assert(block.x < 1024);                                                                                  \
        ScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data, \
                                                                                     (dtype *)qk->data,         \
                                                                                     (dtype *)mask->data,       \
                                                                                     batch_size,                \
                                                                                     head_nums,                 \
                                                                                     q_length,                  \
                                                                                     k_length,                  \
                                                                                     scale);                    \
    }                                                                                                           \
    else                                                                                                        \
    {                                                                                                           \
        constexpr int NUMS_PER_THREAD_PER_ROW = 1;                                                              \
        block.x /= vec_size;                                                                                    \
        assert(block.x < 1024);                                                                                 \
        ScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data, \
                                                                                     (dtype *)qk->data,         \
                                                                                     (dtype *)mask->data,       \
                                                                                     batch_size,                \
                                                                                     head_nums,                 \
                                                                                     q_length,                  \
                                                                                     k_length,                  \
                                                                                     scale);                    \
    }

template <typename T>
void launchScaleMaskAndSoftmax(TensorWrapper<T> *qk,    //输入qk矩阵
                               TensorWrapper<T> *mask,  //输入mask
                               TensorWrapper<T> *attn_score,  //输出softmax结果
                               float scale)   //这个scale就是sqrt(head size) 输不输入都可以
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.  //Output
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.            //Input
    // attention_mask,     (batch_size, q_length, k_length), attention mask.            //Input

    int q_length = qk->shape[2];
    int batch_size = qk->shape[0];
    int head_nums = qk->shape[1];
    int k_length = qk->shape[3];

    bool is_half = sizeof(T) == 2;
    bool is_int8 = sizeof(T) == 1;
    
    if (is_half) {
    	LLM_CHECK_WITH_INFO(k_length % 2 == 0, "Currently, K_len should be divided by 2 under half type!");
    }
    if (is_int8) {
        LLM_CHECK_WITH_INFO(k_length % 4 == 0, "Currently, K_len should be divided by 4 under int8 type!");
    }
    
    dim3 grid(q_length, batch_size, head_nums); 
    dim3 block((k_length + 32 - 1) / 32 * 32);  
    
    if (is_int8)
    {
        // Launch INT8 version - only for int8_t type
        if (block.x > 2048 && block.x <= 4096)
        {
            constexpr int NUMS_PER_THREAD_PER_ROW = 4;
            block.x /= 4 * 4;  // vec_size = 4 for int8
            block.x = (block.x + 32 - 1) / 32 * 32;
            assert(block.x < 1024);
            ScaleMaskAndSoftmax_int8<int8_t, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((int8_t *)attn_score->data,
                                                                                 (int8_t *)qk->data,
                                                                                 (int8_t *)mask->data,
                                                                                 batch_size,
                                                                                 head_nums,
                                                                                 q_length,
                                                                                 k_length,
                                                                                 scale);
        }
        else if (block.x > 1024)
        {
            constexpr int NUMS_PER_THREAD_PER_ROW = 2;
            block.x /= 2 * 4;
            block.x = (block.x + 32 - 1) / 32 * 32;
            assert(block.x < 1024);
            ScaleMaskAndSoftmax_int8<int8_t, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((int8_t *)attn_score->data,
                                                                                 (int8_t *)qk->data,
                                                                                 (int8_t *)mask->data,
                                                                                 batch_size,
                                                                                 head_nums,
                                                                                 q_length,
                                                                                 k_length,
                                                                                 scale);
        }
        else
        {
            constexpr int NUMS_PER_THREAD_PER_ROW = 1;
            block.x /= 4;
            assert(block.x < 1024);
            ScaleMaskAndSoftmax_int8<int8_t, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((int8_t *)attn_score->data,
                                                                                 (int8_t *)qk->data,
                                                                                 (int8_t *)mask->data,
                                                                                 batch_size,
                                                                                 head_nums,
                                                                                 q_length,
                                                                                 k_length,
                                                                                 scale);
        }
    }
    else if (is_half)
    {
        LAUNCH_SOFTMAX(half, 2); 
    }
    else
    {
        LAUNCH_SOFTMAX(float, 1);
    }
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(attn_score->data);
#else
#endif
}

template void launchScaleMaskAndSoftmax(TensorWrapper<float> *qk,
                                        TensorWrapper<float> *mask,
                                        TensorWrapper<float> *attn_score,
                                        float scale);

template void launchScaleMaskAndSoftmax(TensorWrapper<half> *qk,
                                        TensorWrapper<half> *mask,
                                        TensorWrapper<half> *attn_score,
                                        float scale);

template void launchScaleMaskAndSoftmax(TensorWrapper<int8_t> *qk,
                                        TensorWrapper<int8_t> *mask,
                                        TensorWrapper<int8_t> *attn_score,
                                        float scale);
