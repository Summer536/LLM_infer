预计需要两整天的时间来继续解决如下问题：

1. FP32 FP16已经全部实现，部分kernel可能还有些问题，请通过/build/bin/test一一检查。 INT8问题可能比较大，但是也实现了，第二轮再处理吧
2. 使用tools/weights_convert_ture.py（在LLM_GYQ文件夹中）已经成功转换出来了所有的weight.bin文件，位于/home/yqgao/llamaweight/1-gpu/中。但是我不确定它是FP32类型还是FP16类型，需要再次检查
3. 目前整体kernel都搭建完毕了，layer的两个decoder层也搭建完毕了，问题就在/models/llama/llama.cpp中，也就是说对应课程的27节以后的内容，还需要再看一遍
4. 对应解决的问题有：token的编码解码问题？用户输入并解码后如何正确传入llama的整体layer中？user_entry.cpp如何使用

另外需要做如下优化：
1.flashattention （初步实现，后面检查一下）
2.pagedattention
3.contiune batch