# 使用不同的方法构建CUDA库，避免单独的设备链接步骤
# 设置全局CUDA编译选项
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# 禁用默认的单独设备链接
set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE "")
set(CMAKE_CUDA_DEVICE_LINK_LIBRARY "")

# 定义embeddingFunctor库
add_library(embeddingFunctor STATIC input_embedding.cu)
set_target_properties(embeddingFunctor PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(embeddingFunctor PUBLIC -lcudart)

# 定义rmsnorm库
add_library(rmsnorm STATIC rmsnorm_kernel.cu)
set_target_properties(rmsnorm PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(rmsnorm PUBLIC -lcudart)

# 定义cal_paddingoffset库
add_library(cal_paddingoffset STATIC cal_paddingoffset.cu)
set_target_properties(cal_paddingoffset PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cal_paddingoffset PUBLIC -lcudart)

# 定义build_causal_mask库
add_library(build_causal_mask STATIC build_causal_mask.cu)
set_target_properties(build_causal_mask PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(build_causal_mask PUBLIC -lcudart)

# 定义cublasWrapper库
add_library(cublasWrapper STATIC cublas_utils.cc)
set_target_properties(cublasWrapper PROPERTIES 
    POSITION_INDEPENDENT_CODE ON)

# 定义linear库
add_library(linear STATIC linear.cu)
set_target_properties(linear PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(linear PUBLIC -lcudart -lcublas cublasWrapper)

add_library(qkv_bias_and_rope STATIC qkv_bias_and_RoPE.cu)
set_target_properties(qkv_bias_and_rope PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(qkv_bias_and_rope PUBLIC -lcudart)

add_library(concat_kv STATIC concat_past_kv.cu)
set_target_properties(concat_kv PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(concat_kv PUBLIC -lcudart)

add_library(repeat_kv STATIC repeat_kv.cu)
set_target_properties(repeat_kv PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(repeat_kv PUBLIC -lcudart)

add_library(mask_softmax STATIC attn_softmax_kernel.cu)
set_target_properties(mask_softmax PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(mask_softmax PUBLIC -lcudart)

add_library(fused_transpose_and_remv_pad STATIC fused_transpose_and_remv_pad.cu)
set_target_properties(fused_transpose_and_remv_pad PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fused_transpose_and_remv_pad PUBLIC -lcudart)

add_library(fused_addresidual_norm STATIC fused_addresidual_norm.cu)
set_target_properties(fused_addresidual_norm PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fused_addresidual_norm PUBLIC -lcudart)

add_library(act STATIC act_kernel.cu)
set_target_properties(act PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(act PUBLIC -lcudart)

add_library(topK STATIC topK.cu)
set_target_properties(topK PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(topK PUBLIC -lcudart)

add_library(fused_decoder_self_attention STATIC fused_decoder_self_attention.cu)
set_target_properties(fused_decoder_self_attention PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fused_decoder_self_attention PUBLIC -lcudart)

add_library(sampling STATIC sampling.cu)
set_target_properties(sampling PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(sampling PUBLIC -lcudart)

add_library(add_residual STATIC add_residual.cu)
set_target_properties(add_residual PROPERTIES 
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
target_link_libraries(add_residual PUBLIC -lcudart)