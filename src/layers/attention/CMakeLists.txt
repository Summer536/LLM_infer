cmake_minimum_required(VERSION 3.8)
# 编译为静态库
add_library(LlamaCtxAttn STATIC context_attention.cpp)
set_property(TARGET LlamaCtxAttn PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(LlamaCtxAttn PUBLIC
                             -lcudadevrt
                             qkv_bias_and_rope
                             concat_kv
                             linear
                             fused_transpose_and_remv_pad
                             repeat_kv
                             mask_softmax)
# 编译为静态库
add_library(LlamaselfAttn STATIC masked_self_attention.cpp)
set_property(TARGET LlamaselfAttn PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(LlamaselfAttn PUBLIC
                             -lcudart
                             -lcudadevrt
                             fused_decoder_self_attention
                             qkv_bias_and_rope
                             linear
)
