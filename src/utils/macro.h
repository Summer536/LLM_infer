#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Add INT8 quantization support
#define INT8_SCALE_FACTOR 127.0f
#define INT8_INV_SCALE_FACTOR (1.0f / 127.0f)

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

static const char* _cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result) {
        throw std::runtime_error(std::string("[TM][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

#define CHECK_CUBLAS(val) check((val), #val, __FILE__, __LINE__)

inline void syncAndCheck(const char* const file, int const line)
{
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[TM][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

#define DeviceSyncAndCheckCudaError() syncAndCheck(__FILE__, __LINE__) //定位到__FILE__, __LINE__ 哪个文件的哪一行发生bug     //(Lesson30):这个需要放到layer中，因为这个是对一整个kernel的检查，需放在kernel和kernel之间，而不是kernel之内

[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw std::runtime_error(std::string("[oneLLM][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}

inline void llmAssert(bool result, const char* const file, int const line, std::string const& info = "") //断言函数
{
    if (!result) {
        throwRuntimeError(file, line, info);
    }
}

//防御性检查，检查tensor是在gpu上还是cpu上
//在可能出错的地方加上这些check
#define LLM_CHECK(val) llmAssert(val, __FILE__, __LINE__)
#define LLM_CHECK_WITH_INFO(val, info)                                                                              \
    do {                                                                                                               \
        bool is_valid_val = (val);                                                                                     \
        if (!is_valid_val) {                                                                                           \
            llmAssert(is_valid_val, __FILE__, __LINE__, (info));                                                    \
        }                                                                                                              \
    } while (0)
