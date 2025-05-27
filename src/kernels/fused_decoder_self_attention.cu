#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_decoder_self_attention.h"
// kv cache shape = [numlayers, bs, kv head num, max_seq_len, head size]
// bug1: scale's dtype must be float ,not int
// bug2: mha_kernel_params struct's pointer is on CPU, not GPU, which cause we dont run the cuda kernel, so add cudacheck is a must!
// bug3: blockreduce res should use tid=0 to write into smem
// bug4: GQA, kv_head_num brd to head_num, we can automaticly do this by head id index like lmdeploy
// half or float version: the logits and mha output both are fp32 type, q k v are all accessed vectorizedly
template<typename T>
__device__ T warpReduceSum(T val){

    for(int mask = 16; mask > 0; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;

}
template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    //note: here return val of warpreducesum should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceSum<T>(warp_val);

}
template<typename T>
__device__ T warpReduceMax(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpmax[64];
    // returned val is the max computed by 0th thread.
    val = warpReduceMax(val); // remove <T> can ignore the multi-overloaded error?
    //note: here return val of warpreducemax should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpmax[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpmax[tid] : (T)0;
    return warpReduceMax(warp_val);
}

inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    // (RussWong) note: 每个token所属的id, 它的freq值都是固定的, id的上限为max position embedding
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim); //rot_embed_dim = 128
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * data - coef.y * data_rotate;
    rot_v.y = coef.x * data_rotate + coef.y * data;
    return rot_v;
}

// block and thread allocation
// 1 block -> head size，后续可改进为1 warp -> 1 head size or 1 block -> multi head size
// 1 grid -> bs * num heads
// q; input vec [bs, q num heads, 1, head size]
// k; input vec [bs, kv num heads, 1, head size]
// v; input vec [bs, kv num heads, 1, head size]
// k_cache; output,[num layers, bs, kv num heads, max_seq_len or step, head size] from prompt phase
// k_cache; output,[num layers, bs, kv num heads, max_seq_len or step, head size] from prompt phase
template<typename T>
__global__ void masked_MHA_kernel(T* q, //输入: qkv_buf中的q向量
                    T* k, //输入: qkv_buf中的k向量
                    T* v, //输入: qkv_buf中的v向量
                    T* qkv_bias, //输入: qkv向量的bias
                    T* k_cache,  //输入: k_cache
                    T* v_cache,  //输入: v_cache
                    T* mha_output, //输出: 该融合算子的最终结果
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int max_seq_len,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim, //输入: 旋转编码的维度
                    float rotary_embedding_base //输入: 旋转编码的base
                    ){// rsqrt(dh)

    //分配GirdDIm(一维的head_num*batch_size个block)  分配BlockDIm(一维head_size个thread)
    int tid = threadIdx.x;
    int q_batch_id = blockIdx.x / head_num;
    int q_head_id = blockIdx.x % head_num;

    // (RussWong) note: below one line is wrong kv head id access way.
    //int kv_head_id = bid % kv_head_num;
    int kv_head_id = q_head_id / (head_num / kv_head_num); //由于在GQA/MQA中，kv headnum的数量是小于q headnum的，因此这里取kv head的id时，需要对q_head_id除一个(head_num / kv_head_num),()中的内容表示qheadnum比kvheadnum的倍数
    int kv_batch_id = q_batch_id;   //对于batchid 来讲，kv的bs数和q的bs数是一样的

    //求一些偏移量（对q k v三个矩阵求的偏移量），这个stride和pytorch里面的stride是一样的：Stride 是指在内存中从一个元素移动到下一个元素所需的步长（步数）。在多维数组或张量中，不同维度有不同的 stride 值。
    int batch_stride = head_num * head_size;         //q的batch偏移：  从一个 batch 到下一个 batch 的步长为head_num * head_size
    int kv_batch_stride = kv_head_num * head_size;   //kv的batch偏移： 从一个 batch 到下一个 batch 的步长为 kv_head_num * head_size
    int head_stride = head_size;                     //q的head偏移：移动到下一个 head 需要跨越 head_size 个元素
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;      //q   张量中特定 batch、特定 head、特定线程对应元素的偏移量
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid; //k/v 张量中特定 batch、特定 head、特定线程对应元素的偏移量
    //这里其实本质上还是采用的这样一个公式shape=[bs, head num, head size]时，
    // offset = bs_id * head num * head size + head num_id * head size + head size_id  （上方由于thread分配数量为head size个，因此直接由threadID表示head size_id）
    //还有个问题容易混淆：为什么这里求k_offset得时候没有乘以一个max_seq_len呢？答:以上求偏移量得三个矩阵都是从fused qkv gemm算子得结果传入进来的，它们的维度为：
    //                                                                   q; input vec [bs, q num heads, 1, head size]  k; input vec [bs, kv num heads, 1, head size]
    //                                                                   而在这个之后的KVcachekernel之后，我们才需要从KVcache中取k/v的offset，才需要乘一个max_seq_len(下方这个)

    ///////////////////////////////////以下是求向量化读取时的offset偏置情况/////////////////////////////////////
    int vec_size = Vec<T>::size; //(/src/utils/vectorize_utils.h)中已经定义好了各个不同类型的向量化读取的大小值，是一个静态成员变量size, 直接提取它用即可。 例如：如果定义向量化读取float4 则这个size=4
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;

    //同样的，这里也是求一些偏移量（是对kv cache求的偏移量）
    int cache_offset = kv_batch_id * kv_head_num * max_seq_len * head_size +     // k_cache; output,[num layers, bs, kv num heads, max_seq_len or step, head size] 这里的num layers已经在launch的时候确定好了
                        kv_head_id * max_seq_len * head_size + tid * vec_size;   //求解这个cache offset时，因为此时我们需要从kv cache中取k/v，因此需要乘max_seq_len
    int step_stride = head_size;

    float scale = rsqrt(float(head_size)); //这个是那个scale kernel的值，其实特别简单，直接乘以这个值即可

    using Vec_t = typename Vec<T>::Type;
    Vec_t qvec, kvec, vvec; //向量化类型的寄存器
    const T* q_mem = q; //q k v 重命名一下，看起来像是从显存里面去拿的，不然单一个q 看着像寄存器里的值（其实不用这步的hhhhhhh）
    const T* k_mem = k;
    const T* v_mem = v;

    /////////////////////////////////////////////////////////////////////////Add bias////////////////////////////////////////////////////////////////////
    if (tid * vec_size < head_size) {//确保线程访问不要越界
        qvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&q_mem[q_offset_vec])); //这是q矩阵  

        ///////////////////下方注释掉是因为llama2中没有bias，如果在别的大模型中需要，则把它们的注释取消//////////////////////////////
        // (RussWong) note: will enable below code lines when qkv bias is available
        // if (qkv_bias != nullptr){                                                                         //qkv_bias 的shape=[qkv head num, head size]
	    //     Vec_t q_bias = *reinterpret_cast<Vec_t*>(&qkv_bias[q_head_id * head_size + tid * vec_size]);  //这一步是要提取bias值,通过求其offset拿到qkv_bias大矩阵中关于q得bias。 offset = q head id * head size + head size id  
        //     for(int i = 0; i < vec_size; i++) {
        //         reinterpret_cast<float*>(&qvec)[i] += reinterpret_cast<float*>(&q_bias)[i]; //为什么做标量得相加计算，而不是向量呢？因为GPU不支持向量加法hhhhhh  具体做法：&qvec变为指针，再将其强制转换为float类型的指针，通过[i]取对应的值
        //     }
	    // }
        ///////////////////上方注释掉是因为llama2中没有bias，如果在别的大模型中需要，则把它们的注释取消//////////////////////////////

        kvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&k_mem[k_offset_vec])); //这是k矩阵         
        vvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&v_mem[k_offset_vec])); //这是v矩阵  
        ///////////////////下方注释掉是因为llama2中没有bias，如果在别的大模型中需要，则把它们的注释取消//////////////////////////////
        // (RussWong) note: will enable below code lines when qkv bias is available
        // if (qkv_bias != nullptr){
	    //     Vec_t v_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]);
        //     for(int i = 0; i < vec_size; i++) {
        //         reinterpret_cast<float*>(&vvec)[i] += reinterpret_cast<float*>(&v_bias)[i];
        //     }
	    // }
        ///////////////////上方注释掉是因为llama2中没有bias，如果在别的大模型中需要，则把它们的注释取消//////////////////////////////
    }
     /////////////////////////////////////////////////////////////////////////Add bias////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////  RoPE  ////////////////////////////////////////////////////////////////////
    ////////复用了(/sec/kernels/qkv_bias_and_RoPE.cu)的RoPEkernel了，因为llama2的RoPE特性 无法在实现融合////
    /////////////////////////////////////////////////////////////////////////  RoPE  ////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////  Q*k batchGemm  ////////////////////////////////////////////////////////////////////

    extern __shared__ char sqk[]; 
    T* sq_scalar = reinterpret_cast<T*>(sqk); 
    float* logits = reinterpret_cast<float*>(sq_scalar + head_size); 
    Vec_t* sq = reinterpret_cast<Vec_t*>(sq_scalar);
    if (tid * vec_size < head_size) {
        sq[tid] = qvec;  
    }
    __syncthreads();  

    float zero = 0.0f;
    Vec_t zero_f4 = scalar_cast_vec<Vec_t, T>(zero);          
    float4 scale_f4 = scalar_cast_vec<float4, float>(scale);  

    /////////这里没有采用多个block并行处理多个行的乘法，而是采用一个block循环处理多个行，即先处理完第一行再处理第二行...  。这里可以优化的！！！！！！！！！！///////////////
    for(int iter = 0; iter < step; iter++) {
        Vec_t kvec_qk = tid * vec_size < head_size ? *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) : zero_f4;  
        if (iter == step - 1 && tid * vec_size < head_size) {
            *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) = kvec; 
            kvec_qk = kvec; 
        }

        Vec_t qk = zero_f4; 
        qk.x = (tid * vec_size < head_size) ? sq[tid].x * kvec_qk.x * scale_f4.x : zero;  
        qk.y = (tid * vec_size < head_size) ? sq[tid].y * kvec_qk.y * scale_f4.y : zero;
        qk.z = (tid * vec_size < head_size) ? sq[tid].z * kvec_qk.z * scale_f4.z : zero;
        qk.w = (tid * vec_size < head_size) ? sq[tid].w * kvec_qk.w * scale_f4.w : zero;

        T qk_acc = qk.x + qk.y + qk.z + qk.w;

        T attn_score = blockReduceSum<T>(qk_acc);
        if(tid == 0) { 
            logits[iter] = attn_score;     
	    }
        __syncthreads(); 
    }
    ////////////////////////////////////////////////////////////////////  Q*k batchGemm  ////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////  softmax   //////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////   softmax(x_i) = e^(x_i - max(x_i)) / [sum ( e^(x_j - max(x_j)) ) ]  //////////////////////////////////////////
    //softmax(logits), logits.shape = [bs, num heads, 1, step]
    T local_logits = tid < step ? (T)logits[tid] : 0;  
    __shared__ float row_max, fenmu;
    
    T block_max = blockReduceMax<T>(local_logits);
    if (tid == 0){
        row_max = block_max;
    }
    __syncthreads();

    T fenzi = tid < step ? expf(logits[tid] - row_max) : 0;
    
    T block_fenmu = blockReduceSum<T>(fenzi);
    if (tid == 0){
        fenmu = block_fenmu + 1e-6; //1e-6加不加都行
    }
    __syncthreads();

    //求出softmax最终值
    if(tid < step) {
        logits[tid] = (T)(fenzi / fenmu);
    }
    __syncthreads();
    ////////////////////////////////////////////////////////////////////  softmax   //////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////  qk*v batchGemm  ////////////////////////////////////////////////////////////////////
    if (tid * vec_size < head_size) {
        Vec_t O = scalar_cast_vec<Vec_t, T>(0.0f); 
        for(int iter = 0; iter < step; iter++) {
            Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]);
	    if (iter == step - 1) {
                *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]) = vvec;  
                vvec_qkv = vvec;
            }

            O.x += vvec_qkv.x * logits[iter];
            O.y += vvec_qkv.y * logits[iter];
            O.z += vvec_qkv.z * logits[iter];
            O.w += vvec_qkv.w * logits[iter];
        }
        ///////////////////////////////////////////////////////////将最终结果写回到输出结果矩阵中////////////////////////////////////////////////////
        *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = O; //shape = [bs, q_head_num, 1, head size]
    }
    ////////////////////////////////////////////////////////////////////  qk*v batchGemm  ////////////////////////////////////////////////////////////////////
}


template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,    //输入: qkv的buffer，来自于(/src/kernels/linear.cu)中qkv gemm的输出
                            BaseWeight<T>& qkv,           //输入: qkv linear中weights附带的bias，虽然llama2中没有，但是这里加上为了统一接口
                            TensorWrapper<int>* layer_id, //输入: layerid表示当前在第几层
                            TensorWrapper<T>* k_cache,    //输入: k cache   //shape=[num layers, bs, kv head num, max seq len, head size]
                            TensorWrapper<T>* v_cache,    //输入: v cache   //shape=[num layers, bs, kv head num, max seq len, head size]
                            TensorWrapper<bool>* finished,//输入: finished是bool值，表示你当前生成的Token是否是最后一个
                            TensorWrapper<int>* step,     //输入: step表示当前生成的Token的长度，即当前生成到第几个token了，第几步了
                            TensorWrapper<T>* mha_output, //输出: mha_output是输出的buffer,存储当前这个融合算子的输出值
                            LLaMAAttentionStaticParams& static_params){ //输入: static_params记载的有关于Rope的配置

    //qkv_buf shape = [bs, qkv_head_num, head_size]
    //k_cache shape = [num layers, bs, kv_head_num, max_seq_len, head_size]
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2];
    const int max_seq_len = k_cache->shape[3]; 
    const int head_size = qkv_buf->shape[2];

    int head_num = qkv_head_num - 2 * kv_head_num;

    const int cur_step = step->getVal();
    const int layer = layer_id->getVal();

    const int layer_offset = layer * max_seq_len * batch_size * kv_head_num * head_size;

    size_t smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float);

    T* qkv_data = qkv_buf->data;
    //qkv_data.shape = [bs, 1, qkv_head_num,head_size]
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size; //q矩阵的大小:head_num * head_size
    T* v = qkv_data + (head_num + kv_head_num) * head_size; //q和v矩阵拼接起来的大小:(head_num + kv_head_num) * head_size

    //LLaMAAttentionStaticParams& static_params中记载的有关于Rope的配置
    int   rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int   max_position_embeddings = static_params.max_position_embeddings;
    bool  use_dynamic_ntk = static_params.use_dynamic_ntk;

    dim3 grid(head_num * batch_size);
    dim3 block(head_size); //vec size = 4 for fp32 

    //可以输入三个参数到分配GPU，第三个参数smem_size_bytes表示我们要分配这么大的动态share memory
    masked_MHA_kernel<T><<<grid, block, smem_size_bytes>>>(q,
                                                            k,
                                                            v,
                                                            /*(T*)*/qkv.bias,
                                                            k_cache->data + layer_offset,
                                                            v_cache->data + layer_offset,
                                                            mha_output->data,
                                                            batch_size,
                                                            head_num,
                                                            kv_head_num,
                                                            max_seq_len,
                                                            head_size,
                                                            cur_step,
                                                            rotary_embedding_dim,
                                                            rotary_embedding_base);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(mha_output->data, true);
#else
#endif
}


//////////////////////////////////这个kernel是FP16类型的融合自注意力///////////////////////////////////////
template<>
__global__ void masked_MHA_kernel<half>(half* q, //输入: qkv_buf中的q向量
                    half* k, //输入: qkv_buf中的k向量
                    half* v, //输入: qkv_buf中的v向量
                    half* qkv_bias, //输入: qkv向量的bias
                    half* k_cache,  //输入: k_cache
                    half* v_cache,  //输入: v_cache
                    half* mha_output, //输出: 该融合算子的最终结果
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int max_seq_len,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim, //输入: 旋转编码的维度
                    float rotary_embedding_base //输入: 旋转编码的base
                    ){

    //分配GirdDIm(一维的head_num*batch_size个block)  分配BlockDIm(一维head_size个thread)
    int tid = threadIdx.x;
    int q_batch_id = blockIdx.x / head_num;
    int q_head_id = blockIdx.x % head_num;

    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;

    //求一些偏移量
    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;

    ///////////////////////////////////以下是求向量化读取时的offset偏置情况/////////////////////////////////////
    int vec_size = Vec<half>::size; // = 2 for half2
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;

    //同样的，这里也是求一些偏移量（是对kv cache求的偏移量）
    int cache_offset = kv_batch_id * kv_head_num * max_seq_len * head_size +
                        kv_head_id * max_seq_len * head_size + tid * vec_size;
    int step_stride = head_size;

    float scale = rsqrt(float(head_size));

    using Vec_t = typename Vec<half>::Type; // half2
    Vec_t qvec, kvec, vvec;
    const half* q_mem = q;
    const half* k_mem = k;
    const half* v_mem = v;

    /////////////////////////////////////////////////////////////////////////Add bias////////////////////////////////////////////////////////////////////
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&q_mem[q_offset_vec]));
        kvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&k_mem[k_offset_vec]));
        vvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&v_mem[k_offset_vec]));
    }

    ////////////////////////////////////////////////////////////////////  Q*k batchGemm  ////////////////////////////////////////////////////////////////////
    extern __shared__ char sqk[];
    half* sq_scalar = reinterpret_cast<half*>(sqk);
    float* logits = reinterpret_cast<float*>(sq_scalar + head_size);
    Vec_t* sq = reinterpret_cast<Vec_t*>(sq_scalar);
    if (tid * vec_size < head_size) {
        sq[tid] = qvec;
    }
    __syncthreads();

    half zero_h = __float2half(0.0f);
    Vec_t zero_h2 = scalar_cast_vec<Vec_t, half>(zero_h);
    half2 scale_h2 = scalar_cast_vec<half2, half>(__float2half(scale));

    for(int iter = 0; iter < step; iter++) {
        Vec_t kvec_qk = tid * vec_size < head_size ? *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) : zero_h2;
        if (iter == step - 1 && tid * vec_size < head_size) {
            *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) = kvec;
            kvec_qk = kvec;
        }

        // FP16 vectorized computation
        Vec_t qk = __hmul2(__hmul2(sq[tid], kvec_qk), scale_h2);
        
        // Sum across vector elements - convert to float for accumulation
        float qk_acc = __half2float(qk.x) + __half2float(qk.y);

        float attn_score = blockReduceSum<float>(qk_acc);
        if(tid == 0) {
            logits[iter] = attn_score;
        }
        __syncthreads();
    }

    ////////////////////////////////////////////////////////////////////  softmax   //////////////////////////////////////////////////////////////////////////
    float local_logits = tid < step ? logits[tid] : 0.0f;
    __shared__ float row_max, fenmu;
    
    float block_max = blockReduceMax<float>(local_logits);
    if (tid == 0){
        row_max = block_max;
    }
    __syncthreads();

    float fenzi = tid < step ? expf(logits[tid] - row_max) : 0.0f;
    
    float block_fenmu = blockReduceSum<float>(fenzi);
    if (tid == 0){
        fenmu = block_fenmu + 1e-6f;
    }
    __syncthreads();

    //求出softmax最终值
    if(tid < step) {
        logits[tid] = fenzi / fenmu;
    }
    __syncthreads();

    ////////////////////////////////////////////////////////////////////  qk*v batchGemm  ////////////////////////////////////////////////////////////////////
    if (tid * vec_size < head_size) {
        Vec_t O = scalar_cast_vec<Vec_t, half>(__float2half(0.0f));
        for(int iter = 0; iter < step; iter++) {
            Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]);
            if (iter == step - 1) {
                *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]) = vvec;
                vvec_qkv = vvec;
            }

            // FP16 multiply-accumulate
            half logit_h = __float2half(logits[iter]);
            half2 logit_h2 = scalar_cast_vec<half2, half>(logit_h);
            O = __hfma2(vvec_qkv, logit_h2, O); // O += vvec_qkv * logit
        }
        ///////////////////////////////////////////////////////////将最终结果写回到输出结果矩阵中////////////////////////////////////////////////////
        *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = O;
    }
}

//////////////////////////////////这个kernel是INT8类型的融合自注意力///////////////////////////////////////
template<>
__global__ void masked_MHA_kernel<int8_t>(int8_t* q, //输入: qkv_buf中的q向量
                    int8_t* k, //输入: qkv_buf中的k向量
                    int8_t* v, //输入: qkv_buf中的v向量
                    int8_t* qkv_bias, //输入: qkv向量的bias
                    int8_t* k_cache,  //输入: k_cache
                    int8_t* v_cache,  //输入: v_cache
                    int8_t* mha_output, //输出: 该融合算子的最终结果
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int max_seq_len,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim, //输入: 旋转编码的维度
                    float rotary_embedding_base //输入: 旋转编码的base
                    ){

    //分配GirdDIm(一维的head_num*batch_size个block)  分配BlockDIm(一维head_size个thread)
    int tid = threadIdx.x;
    int q_batch_id = blockIdx.x / head_num;
    int q_head_id = blockIdx.x % head_num;

    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;

    //求一些偏移量
    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;

    ///////////////////////////////////以下是求向量化读取时的offset偏置情况/////////////////////////////////////
    int vec_size = Vec<int8_t>::size; // = 4 for char4
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;

    //同样的，这里也是求一些偏移量（是对kv cache求的偏移量）
    int cache_offset = kv_batch_id * kv_head_num * max_seq_len * head_size +
                        kv_head_id * max_seq_len * head_size + tid * vec_size;
    int step_stride = head_size;

    float scale = rsqrt(float(head_size));

    using Vec_t = typename Vec<int8_t>::Type; // char4
    Vec_t qvec, kvec, vvec;
    const int8_t* q_mem = q;
    const int8_t* k_mem = k;
    const int8_t* v_mem = v;

    /////////////////////////////////////////////////////////////////////////Add bias////////////////////////////////////////////////////////////////////
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t*>(const_cast<int8_t*>(&q_mem[q_offset_vec]));
        kvec = *reinterpret_cast<Vec_t*>(const_cast<int8_t*>(&k_mem[k_offset_vec]));
        vvec = *reinterpret_cast<Vec_t*>(const_cast<int8_t*>(&v_mem[k_offset_vec]));
    }

    ////////////////////////////////////////////////////////////////////  Q*k batchGemm  ////////////////////////////////////////////////////////////////////
    extern __shared__ char sqk[];
    int8_t* sq_scalar = reinterpret_cast<int8_t*>(sqk);
    float* logits = reinterpret_cast<float*>(sq_scalar + head_size);
    Vec_t* sq = reinterpret_cast<Vec_t*>(sq_scalar);
    if (tid * vec_size < head_size) {
        sq[tid] = qvec;
    }
    __syncthreads();

    int8_t zero_i8 = 0;
    Vec_t zero_i8_4 = scalar_cast_vec<Vec_t, int8_t>(zero_i8);

    for(int iter = 0; iter < step; iter++) {
        Vec_t kvec_qk = tid * vec_size < head_size ? *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) : zero_i8_4;
        if (iter == step - 1 && tid * vec_size < head_size) {
            *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) = kvec;
            kvec_qk = kvec;
        }

        // INT8 computation: dequantize, compute, accumulate in float
        float qk_acc = 0.0f;
        if (tid * vec_size < head_size) {
            // Dequantize and compute Q*K
            float qx = (float)sq[tid].x * INT8_INV_SCALE_FACTOR;
            float qy = (float)sq[tid].y * INT8_INV_SCALE_FACTOR;
            float qz = (float)sq[tid].z * INT8_INV_SCALE_FACTOR;
            float qw = (float)sq[tid].w * INT8_INV_SCALE_FACTOR;
            
            float kx = (float)kvec_qk.x * INT8_INV_SCALE_FACTOR;
            float ky = (float)kvec_qk.y * INT8_INV_SCALE_FACTOR;
            float kz = (float)kvec_qk.z * INT8_INV_SCALE_FACTOR;
            float kw = (float)kvec_qk.w * INT8_INV_SCALE_FACTOR;
            
            qk_acc = (qx * kx + qy * ky + qz * kz + qw * kw) * scale;
        }

        float attn_score = blockReduceSum<float>(qk_acc);
        if(tid == 0) {
            logits[iter] = attn_score;
        }
        __syncthreads();
    }

    ////////////////////////////////////////////////////////////////////  softmax   //////////////////////////////////////////////////////////////////////////
    float local_logits = tid < step ? logits[tid] : 0.0f;
    __shared__ float row_max, fenmu;
    
    float block_max = blockReduceMax<float>(local_logits);
    if (tid == 0){
        row_max = block_max;
    }
    __syncthreads();

    float fenzi = tid < step ? expf(logits[tid] - row_max) : 0.0f;
    
    float block_fenmu = blockReduceSum<float>(fenzi);
    if (tid == 0){
        fenmu = block_fenmu + 1e-6f;
    }
    __syncthreads();

    //求出softmax最终值
    if(tid < step) {
        logits[tid] = fenzi / fenmu;
    }
    __syncthreads();

    ////////////////////////////////////////////////////////////////////  qk*v batchGemm  ////////////////////////////////////////////////////////////////////
    if (tid * vec_size < head_size) {
        // Compute in float, then quantize back
        float Ox = 0.0f, Oy = 0.0f, Oz = 0.0f, Ow = 0.0f;
        
        for(int iter = 0; iter < step; iter++) {
            Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]);
            if (iter == step - 1) {
                *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]) = vvec;
                vvec_qkv = vvec;
            }

            // Dequantize V and accumulate
            float vx = (float)vvec_qkv.x * INT8_INV_SCALE_FACTOR;
            float vy = (float)vvec_qkv.y * INT8_INV_SCALE_FACTOR;
            float vz = (float)vvec_qkv.z * INT8_INV_SCALE_FACTOR;
            float vw = (float)vvec_qkv.w * INT8_INV_SCALE_FACTOR;
            
            float logit_weight = logits[iter];
            Ox += vx * logit_weight;
            Oy += vy * logit_weight;
            Oz += vz * logit_weight;
            Ow += vw * logit_weight;
        }
        
        // Quantize back to INT8
        Vec_t O;
        O.x = (int8_t)__float2int_rn(Ox * INT8_SCALE_FACTOR);
        O.y = (int8_t)__float2int_rn(Oy * INT8_SCALE_FACTOR);
        O.z = (int8_t)__float2int_rn(Oz * INT8_SCALE_FACTOR);
        O.w = (int8_t)__float2int_rn(Ow * INT8_SCALE_FACTOR);
        
        ///////////////////////////////////////////////////////////将最终结果写回到输出结果矩阵中////////////////////////////////////////////////////
        *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = O;
    }
}

template void launchDecoderMaskedMHA(TensorWrapper<float>* qkv_buf,
                            BaseWeight<float>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<float>* k_cache,
                            TensorWrapper<float>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<float>* mha_output,
                            LLaMAAttentionStaticParams& static_params);

template void launchDecoderMaskedMHA(TensorWrapper<half>* qkv_buf,
                            BaseWeight<half>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<half>* k_cache,
                            TensorWrapper<half>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<half>* mha_output,
                            LLaMAAttentionStaticParams& static_params);

template void launchDecoderMaskedMHA(TensorWrapper<int8_t>* qkv_buf,
                            BaseWeight<int8_t>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<int8_t>* k_cache,
                            TensorWrapper<int8_t>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<int8_t>* mha_output,
                            LLaMAAttentionStaticParams& static_params);

