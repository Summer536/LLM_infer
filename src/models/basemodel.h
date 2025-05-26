#pragma once
#include <string>
#include <functional>
#include "src/utils/tensor.h"
#include "src/models/common_params.h"
#include "src/memory/allocator/base_allocator.h"
#include "src/kernels/cublas_utils.h"

using CallBack = std::function<void(int index, const char* GenerateContent)>;

class BaseModel{
public:
    std::string model_name; 

    cudaStream_t stream;
    cublasWrapper* cublas_wrapper;
    BaseAllocator* allocator;
    cudaDeviceProp* cuda_device_prop;

    BaseModel(cudaStream_t stream, 
              cublasWrapper* cublas_wrapper,
              BaseAllocator* allocator,
              cudaDeviceProp* cuda_device_prop = nullptr):
        stream(stream),
        cublas_wrapper(cublas_wrapper),
        allocator(allocator),
        cuda_device_prop(cuda_device_prop){};
    

    virtual void loadTokenizer(std::string file) = 0;
    virtual void loadWeights(std::string file) = 0;
    virtual void loadWeightsFromDummy() = 0;

    virtual std::vector<std::string> MakeInput(const std::string &history, int round, const std::string &input) = 0;

    virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) = 0;

    virtual std::string Response(const std::vector<std::string>& input, CallBack PrintRes) = 0;
};
