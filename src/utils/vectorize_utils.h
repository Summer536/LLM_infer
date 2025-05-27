#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>

template <typename T_OUT, typename T_IN>
inline __device__ T_OUT scalar_cast_vec(T_IN val){
    return val;
}

template <>
inline __device__ half2 scalar_cast_vec<half2, float>(float val){
    return __float2half2_rn(val);
}

template <>
inline __device__ float4 scalar_cast_vec<float4, float>(float val){
    return make_float4(val, val, val, val);
}

template <>
inline __device__ float2 scalar_cast_vec<float2, float>(float val){
    return make_float2(val, val);
}

template <>
inline __device__ half2 scalar_cast_vec<half2, half>(half val){
    half2 res;
    res.x = val;
    res.y = val;
    return res;
}

// Add INT8 scalar cast
template <>
inline __device__ char4 scalar_cast_vec<char4, int8_t>(int8_t val){
    return make_char4(val, val, val, val);
}

template <>
inline __device__ char2 scalar_cast_vec<char2, int8_t>(int8_t val){
    return make_char2(val, val);
}

template <typename T>
struct Vec {
    using Type = T;
    static constexpr int size = 0;
};

template <>
struct Vec<half> {
    using Type = half2;
    static constexpr int size = 2;
};

template <>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};

// Add INT8 vectorization support
template <>
struct Vec<int8_t> {
    using Type = char4;
    static constexpr int size = 4;
};

// struct TwoFloat2{
//     float2 x;
//     float2 y;
// };

