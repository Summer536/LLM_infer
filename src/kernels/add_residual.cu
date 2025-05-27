#include <stdio.h>
#include "src/kernels/add_residual.h"
#include "src/utils/cuda_debug_utils.cuh"

//////////////////////////////////这个kernel是FP32类型的残差加///////////////////////////////////////
template <typename T>
__global__ void AddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    T *residual,
    T *decoder_out, // [num tokens, hidden_units]
    int num_tokens,
    int hidden_units)
{   
    ///////////由于我们这个kernel本质上是一个element width类型的计算，因此我们如下采用float4向量化读取的方式来计算，以提高带宽利用率。/////////

    // 获取数据类型T的向量大小和向量类型
    int vec_size = Vec<T>::size;   //位于(/src/utils/vectorize_utils.h中)
    using Vec_t = typename Vec<T>::Type;

    //分配grid维度为dim1，bs个； 分配block维度为dim1，256个 
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;

    Vec_t *dout = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);

    // 保证256个线程能够遍历完一行hidden_units个数据，/vec_size是因为一个线程要向量化读取 vec_size 个数据
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x)
    {
        // 残差加，对于fp32，只能标量计算
        dout[i].x += rsd[i].x;
        dout[i].y += rsd[i].y;
        dout[i].z += rsd[i].z;
        dout[i].w += rsd[i].w;
    } // addresidual
}

//////////////////////////////////这个kernel是FP16类型的残差加///////////////////////////////////////
template <>
__global__ void AddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    half *residual,
    half *decoder_out, // [num tokens, hidden_units]
    int num_tokens,
    int hidden_units)
{
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *dout = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x)
    {
        dout[i] = __hadd2(dout[i], rsd[i]); 
    } // addresidual
}

//////////////////////////////////这个kernel是INT8类型的残差加///////////////////////////////////////
template <>
__global__ void AddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    int8_t *residual,
    int8_t *decoder_out, // [num tokens, hidden_units]
    int num_tokens,
    int hidden_units)
{
    int vec_size = Vec<int8_t>::size;
    using Vec_t = typename Vec<int8_t>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *dout = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);
    
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x)
    {
        // For INT8, we need to dequantize, add, then quantize back
        // Dequantize to float for computation
        float dx = (float)dout[i].x * INT8_INV_SCALE_FACTOR;
        float dy = (float)dout[i].y * INT8_INV_SCALE_FACTOR;
        float dz = (float)dout[i].z * INT8_INV_SCALE_FACTOR;
        float dw = (float)dout[i].w * INT8_INV_SCALE_FACTOR;
        
        float rx = (float)rsd[i].x * INT8_INV_SCALE_FACTOR;
        float ry = (float)rsd[i].y * INT8_INV_SCALE_FACTOR;
        float rz = (float)rsd[i].z * INT8_INV_SCALE_FACTOR;
        float rw = (float)rsd[i].w * INT8_INV_SCALE_FACTOR;
        
        // Add in float precision
        dx += rx;
        dy += ry;
        dz += rz;
        dw += rw;
        
        // Quantize back to int8
        dout[i].x = (int8_t)__float2int_rn(dx * INT8_SCALE_FACTOR);
        dout[i].y = (int8_t)__float2int_rn(dy * INT8_SCALE_FACTOR);
        dout[i].z = (int8_t)__float2int_rn(dz * INT8_SCALE_FACTOR);
        dout[i].w = (int8_t)__float2int_rn(dw * INT8_SCALE_FACTOR);
    } // addresidual
}

template <typename T>
void launchAddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, 256 threads travrse hiddenunits eles recursely
    TensorWrapper<T> *residual, //输入： [num tokens, hidden_units]（这里的第一维仅仅是在contextdecoder阶段是num tokens，在selfdecoder阶段是batchsize！）
                                                                    //再次提醒：在selfdecoder阶段我们的输入token数永远为1，也就是seqlen=1，所以num tokens = batchsize * 1
    TensorWrapper<T> *decoder_out, //输入兼输出： [num tokens, hidden_units]
    //这里residual和decoder_out的维度是一样的，因此我们可以直接做点对点的操作。  那么如果假设它俩不一样呢？怎么办？答：做一个广播操作
    bool is_print
)
{   
    int batch_size = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;

    //分配grid维度为dim1，bs个； 分配block维度为dim1，256个 
    dim3 grid(batch_size);
    dim3 block(256);
    AddResidual<T><<<grid, block>>>(residual->data,
                                    decoder_out->data,
                                    batch_size,
                                    hidden_units);
#ifdef PRINT_DATA
    if (is_print){
        print_data<<<1, 1>>>(decoder_out->data);
    }
#else
#endif
}
template void launchAddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    TensorWrapper<float> *residual,
    TensorWrapper<float> *decoder_out, // [num tokens, hidden_units]
    bool is_print
    );
template void launchAddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    TensorWrapper<half> *residual,
    TensorWrapper<half> *decoder_out, // [num tokens, hidden_units]
    bool is_print
    );
template void launchAddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    TensorWrapper<int8_t> *residual,
    TensorWrapper<int8_t> *decoder_out, // [num tokens, hidden_units]
    bool is_print
    );
