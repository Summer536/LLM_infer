#include <iostream>
#include "src/kernels/act_kernel.h"
#include "src/utils/cuda_debug_utils.cuh"
#include "src/utils/macro.h"
// fp32 silu version
template<typename T>
__device__ __forceinline__ T silu(const T& in) {
  // x * sigmoid(x)
  return (T) (((float) in) / (1.0f + expf((float) -in)));
}
// fp16 silu version
template<>
__device__ __forceinline__ half2 silu<half2>(const half2& in) {
  return make_half2(__float2half(silu<float>((float)(in.x))), __float2half(silu<float>((float)(in.y)))); //调用FP32的silu做计算，将传入数据强转为FP32类型，将输出强转为FP16类型。
        //make half2: 将两个half类型的数据组合成一个half2(向量化)类型的变量
}

// int8 silu version - convert to float, compute, then quantize back
template<>
__device__ __forceinline__ char4 silu<char4>(const char4& in) {
  // Dequantize to float, compute silu, then quantize back
  float x = (float)in.x * INT8_INV_SCALE_FACTOR;
  float y = (float)in.y * INT8_INV_SCALE_FACTOR;
  float z = (float)in.z * INT8_INV_SCALE_FACTOR;
  float w = (float)in.w * INT8_INV_SCALE_FACTOR;
  
  x = silu<float>(x);
  y = silu<float>(y);
  z = silu<float>(z);
  w = silu<float>(w);
  
  // Quantize back to int8
  return make_char4(
    (int8_t)__float2int_rn(x * INT8_SCALE_FACTOR),
    (int8_t)__float2int_rn(y * INT8_SCALE_FACTOR),
    (int8_t)__float2int_rn(z * INT8_SCALE_FACTOR),
    (int8_t)__float2int_rn(w * INT8_SCALE_FACTOR)
  );
}

// 这里第一个intermediate表示gate linear后的数据，第二个intermediate表示up linear后的数据
// 代码逻辑：第一个intermediate 去做silu，其结果 与 第二个intermediate 做点乘mul
// silu:  x * sigmoid(x) 
template<typename T>
__global__ void silu_and_mul_kernel(
  T* out,               // shape: [bs, intermedia size]
  const T* input,       // shape: [bs, 2, intermedia size]
  const int intermedia_size) {
  
  const int batch_idx = blockIdx.x;

  for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x) { 

    //shape: [bs, 2, intermedia size]
    const T x = input[batch_idx * 2 * intermedia_size + idx];
    //shape: [bs, 2, intermedia size]
    const T y = input[batch_idx * 2 * intermedia_size + intermedia_size + idx];
    out[batch_idx * intermedia_size + idx] = silu<T>(x) * y;
  }
}

template<>
__global__ void silu_and_mul_kernel<half>(
  half* out,               // [bs, intermedia size]
  const half* input,       // [bs, 2, intermedia size]
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;

  int vec_size = Vec<half>::size;

  using Vec_t = typename Vec<half>::Type;
  for (int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x) {
    const Vec_t x = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * 2 * intermedia_size + idx]));
    const Vec_t y = *reinterpret_cast<Vec_t*>(const_cast<half*>(&input[batch_idx * 2 * intermedia_size + intermedia_size + idx]));
    *reinterpret_cast<Vec_t*>(&out[batch_idx * intermedia_size + idx]) = __hmul2(silu<Vec_t>(x), y); //__hmul2 是FP16的点乘函数
  }
}

template<>
__global__ void silu_and_mul_kernel<int8_t>(
  int8_t* out,               // [bs, intermedia size]
  const int8_t* input,       // [bs, 2, intermedia size]
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;

  int vec_size = Vec<int8_t>::size;

  using Vec_t = typename Vec<int8_t>::Type;
  for (int idx = threadIdx.x * vec_size; idx < intermedia_size; idx += blockDim.x * vec_size) {
    const Vec_t x = *reinterpret_cast<const Vec_t*>(&input[batch_idx * 2 * intermedia_size + idx]);
    const Vec_t y = *reinterpret_cast<const Vec_t*>(&input[batch_idx * 2 * intermedia_size + intermedia_size + idx]);
    
    // For INT8, we need to do element-wise multiplication after silu
    // Dequantize, compute silu, multiply, then quantize back
    Vec_t silu_x = silu<Vec_t>(x);
    
    // Element-wise multiplication for char4 (INT8 vector type)
    char4 result;
    float fx = (float)silu_x.x * INT8_INV_SCALE_FACTOR;
    float fy = (float)silu_x.y * INT8_INV_SCALE_FACTOR;
    float fz = (float)silu_x.z * INT8_INV_SCALE_FACTOR;
    float fw = (float)silu_x.w * INT8_INV_SCALE_FACTOR;
    
    float gx = (float)y.x * INT8_INV_SCALE_FACTOR;
    float gy = (float)y.y * INT8_INV_SCALE_FACTOR;
    float gz = (float)y.z * INT8_INV_SCALE_FACTOR;
    float gw = (float)y.w * INT8_INV_SCALE_FACTOR;
    
    result.x = (int8_t)__float2int_rn((fx * gx) * INT8_SCALE_FACTOR);
    result.y = (int8_t)__float2int_rn((fy * gy) * INT8_SCALE_FACTOR);
    result.z = (int8_t)__float2int_rn((fz * gz) * INT8_SCALE_FACTOR);
    result.w = (int8_t)__float2int_rn((fw * gw) * INT8_SCALE_FACTOR);
    
    *reinterpret_cast<Vec_t*>(&out[batch_idx * intermedia_size + idx]) = result;
  }
}

template<typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out) {
    // Input shape: [bs, 2, intermedia size]
    int batch_size = input->shape[0];
    LLM_CHECK(input->shape[1] == 2);
    int intermedia_size = input->shape[2];

    dim3 grid(batch_size);
    dim3 block(256);
    silu_and_mul_kernel<T><<<grid, block>>>(out->data, input->data, intermedia_size);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(out->data);
#else
#endif
}
// We must instancite the template, if not, will report linking issue
template void launchAct(TensorWrapper<float>* input, TensorWrapper<float>* output);
template void launchAct(TensorWrapper<half>* input, TensorWrapper<half>* output);
template void launchAct(TensorWrapper<int8_t>* input, TensorWrapper<int8_t>* output);
