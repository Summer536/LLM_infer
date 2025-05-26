#pragma once
#include <cuda_runtime.h>
#include <float.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
template<typename T, int K>


struct topK   //校招生考点：struct的成员属性默认是public！而class默认是private
                        //public的意思是，这里面的成员val[K]和id[K]可以被外部直接访问，"而不是"只能被struct中的这两个__device__ void函数访问。后半句话就是private的意思。
{
    T val[K];
    int id[K];

    __device__ void init(){  
        for (int i = 0; i < K; i++) {
            id[i] = -1;  
            val[i] = -1e-20; 
        }
    }

    __device__ void insertHeap(T data, int data_id){
        float v = (float)val[K-1];
        if(id[K-1] == -1 || v < (float)data){ 
            id[K-1] = data_id; 
            val[K-1] = data;
        }
        for (int i = K - 2; i >= 0; i--){ 
            if(val[i + 1] > val[i] || id[i] == -1) { 
                T tmp = val[i];
                val[i] = val[i + 1];
                val[i + 1] = tmp;                
                int tmp_id = id[i];
                id[i] = id[i + 1];
                id[i + 1] = tmp_id;
            }
        }
    }
};


template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T> *probs,
                             TensorWrapper<int> *topk_ids,
                             TensorWrapper<T> *topk_vals,
                             TensorWrapper<int> *final_topk_ids,
                             TensorWrapper<T> *final_topk_vals);