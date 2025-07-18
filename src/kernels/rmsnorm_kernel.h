#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>

#include "src/utils/vectorize_utils.h"
#include "src/utils/tensor.h"
#include "src/weights/llama/norm_weights.h"

template <typename T>
void launchRMSNorm(TensorWrapper<T> * decoder_out,
                   TensorWrapper<T> * decoder_residual,
                   LayerNormWeight<T>& attn_norm_weight,
                   float eps,
                   bool is_last = false);
                               