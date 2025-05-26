#pragma once
#include "src/weights/base_weights.h"
template<typename T>
struct LLaMAattentionWeights {
    // struct BaseWeight { 
    // std::vector<int> shape; 
    // T*   data;  
    // WeightType type; 
    // T*   bias; 
    BaseWeight<T> q;
    BaseWeight<T> k;
    BaseWeight<T> v;
    BaseWeight<T> qkv;
    BaseWeight<T> output;
};
