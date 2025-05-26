#include "src/utils/weight_utils.h"

template<typename T_OUT, typename T_IN>
inline __device__ T_OUT type_cast(T_IN val) { 
    return val;
}
template<>
inline __device__ float type_cast(half val) { 
    return __half2float(val); 
}

template<>
inline __device__ half type_cast(float val) { 
    return __float2half(val); 
}




template<typename T>
void GPUMalloc(T** ptr, size_t size) 
{
    LLM_CHECK_WITH_INFO(size >= ((size_t)0), "Ask cudaMalloc size " + std::to_string(size) + "< 0 is invalid.");  
    CHECK(cudaMalloc((void**)(ptr), sizeof(T) * size));
}
template void GPUMalloc(float** ptr, size_t size); 
template void GPUMalloc(half** ptr, size_t size);  




template<typename T>
void GPUFree(T* ptr)
{
    if (ptr != NULL) {
        CHECK(cudaFree(ptr));
        ptr = NULL;
    }
}
template void GPUFree(float* ptr);
template void GPUFree(half* ptr);




template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const size_t size)
{
    CHECK(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cudaH2Dcpy(float* tgt, const float* src, const size_t size);
template void cudaH2Dcpy(half* tgt, const half* src, const size_t size);





template<typename T_IN, typename T_OUT>
__global__ void type_conversion(T_OUT* dst, const T_IN* src, const int size) //这个是对传入的数据类型进行转换，比如将FP32-->FP16
{
    int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_thread_nums = blockDim.x * gridDim.x;
    for (int index = gtid; index < size; index += total_thread_nums) {
        dst[index] = type_cast<T_OUT>(src[index]); 
    }
}

template<typename T_IN, typename T_OUT>
void cuda_type_conversion(T_OUT* dst, const T_IN* src, const int size)  //这个是对传入的数据类型进行转换，比如将FP32-->FP16
{
    dim3 grid(128);
    dim3 block(128);
    type_conversion<T_IN, T_OUT><<<grid, block, 0, 0>>>(dst, src, size);
}

template void cuda_type_conversion(float* dst, const half* src, const int size);
template void cuda_type_conversion(half* dst, const float* src, const int size);


// from FT code 这个函数原型来自fasttransormer中，从二进制文件中读取数据，如果读取失败则返回一个空指针
// loads data from binary file. If it succeeds, returns a non-empty (shape size) vector. If loading fails or
// the product of the elements in shape is 0, this function will return an empty vector.
template<typename T>
std::vector<T> loadWeightFromBinHelper(std::vector<size_t> shape, std::string filename)
{
    if (shape.size() > 2) {
        printf("[ERROR] shape should have less than two dims \n");
        return std::vector<T>();
    }
    size_t dim0 = shape[0], dim1 = 1;
    if (shape.size() == 2) {
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if (size == 0) {
        std::cout << "shape is zero, skip loading weight from file: " << filename << std::endl;
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream  in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        std::cout << "file" << filename << "cannot be opened, loading model fails!" << std::endl;
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    std::cout << "Read " << std::to_string(loaded_data_size) << " bytes from " << filename << std::endl;
    in.read((char*)host_array.data(), loaded_data_size); //读取文件中的数据，保存到host_array.data()中

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        return std::vector<T>();
    }
    in.close();
    // If we succeed, return an array with values.
    return host_array;
}



template <typename T_OUT, typename T_FILE> //T_FILE表示读取文件中的数据类型，T_OUT表示导出的数据类型。 我们使用时写的是<T,float>，因为我们使用python脚本转换好的weight类型为float
struct loadWeightFromBin<T_OUT, T_FILE, true> //最后一个参数一个为true 一个为flase表示什么呢？如果T_FILE和T_OUT是一样的，那么它为ture，不一样则为flase
{
public:
    static void internalFunc(T_OUT* ptr, std::vector<size_t> shape, std::string filename) {
        std::vector<T_FILE> host_array = loadWeightFromBinHelper<T_FILE>(shape, filename); //调用loadWeightFromBinHelper函数
        if (host_array.empty()) {
            return;
        }

        cudaH2Dcpy(ptr, host_array.data(), host_array.size()); //将host_array（读取到的文件内数据值）copy到指针上去
        return;    
   }
};

template <typename T_OUT, typename T_FILE>
struct loadWeightFromBin<T_OUT, T_FILE, false>
{
public:
    static void internalFunc(T_OUT* ptr, std::vector<size_t> shape, std::string filename) {  //这个时侯T_FILE和T_OUT不一样，因此需要多一些操作！
        std::vector<T_FILE> host_array = loadWeightFromBinHelper<T_FILE>(shape, filename);
        if (host_array.empty()) {
            return;
        }

        //转换过程：
        T_FILE* ptr_tmp;//定义一个T_FILE类型的临时指针
        GPUMalloc(&ptr_tmp, host_array.size()); //分配临时指针内存
        cudaH2Dcpy(ptr_tmp, host_array.data(), host_array.size()); //用临时指针接住读取到的数据
        cuda_type_conversion(ptr, ptr_tmp, host_array.size());  //这一步将临时指针上的数据做类型转换，然后写入我们传入的指针上！ 这个函数在上面定义好了
        GPUFree(ptr_tmp); //free掉临时指针
        return;
    }
};

template struct loadWeightFromBin<float, float, true>;
template struct loadWeightFromBin<half, half, true>;
template struct loadWeightFromBin<float, half, false>;
template struct loadWeightFromBin<half, float, false>;
