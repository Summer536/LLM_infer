#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

class BaseAllocator // unified interface for all derived allocator to alloc buffer
{
public:
    virtual ~BaseAllocator(){};

    ///////////////////////////malloc/cudamalloc//////////////////////

    template<typename T>
    T* Malloc(T* ptr, size_t size, bool is_host){ 
        return (T*)UnifyMalloc((void*)ptr, size, is_host); 
    }

    //纯虚函数
    virtual void* UnifyMalloc(void* ptr, size_t size, bool is_host = false) = 0; 

    ///////////////////////////free/cudafree//////////////////////
    template<typename T>
    void Free(T* ptr, bool is_host = false){    
        UnifyFree((void*)ptr, is_host);
    }
    virtual void UnifyFree(void* ptr, bool is_host = false) = 0;
};
