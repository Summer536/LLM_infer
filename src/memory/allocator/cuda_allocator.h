#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>
#include "src/memory/allocator/base_allocator.h"
#include "src/utils/macro.h"

/////////////////////////////////////////////////////////1. 定义Block(包括Big和small两个)/////////////////////////////////////////////////
struct CudaBigBlock {
    void *data;  
    size_t size; 
    bool is_allocated; 

    CudaBigBlock() = default;  
    CudaBigBlock(void* data_, int size_, bool is_allocated_): 
        data(data_), 
        size(size_),
        is_allocated(is_allocated_){}
};

struct CudaSmallBlock {
    void *data;
    size_t size;
    bool is_allocated;

    CudaSmallBlock() = default;
    CudaSmallBlock(void* data_, int size_, bool is_allocated_):
        data(data_),
        size(size_),
        is_allocated(is_allocated_){}
};


class CudaAllocator: public BaseAllocator {  
private:
    /////////////////////////////////////////////////////////2. 定义Blockpool(包括Big和small两个)/////////////////////////////////////////////////
    //{device id: block}
    std::map<int, std::vector<CudaSmallBlock> > cudaSmallBlocksMap;    
    std::map<int, std::vector<CudaBigBlock> > cudaBigBlocksMap;
    std::map<int, size_t> FreeSize;  
    size_t total_allocated_size = 0;  
    int dev_id; 
public:
    CudaAllocator() {  
        cudaGetDevice(&dev_id); 
    }
    ~CudaAllocator() { 
        for (auto &it: cudaSmallBlocksMap) {
            auto &cudaBlocks = it.second; 
            for (int i = 0; i < cudaBlocks.size(); i++) {  
                cudaFree(cudaBlocks[i].data); 
            }
            auto &bigBlocks = cudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                cudaFree(bigBlocks[i].data);
            }            
        }
    }

    ////////////////////////////3. 定义unifymalloc分配内存(包括1.通过malloc函数直接分配cpu内存 2.从GPU的Bigpool中分配 3.从GPU的smallpool中分配 4.大小池子中都没分配上的话再去使用cudamalloc分配)//////////////////////////

    void* UnifyMalloc(void* ptr, size_t size, bool is_host) { 
        // 1. host malloc
        if (is_host) { 
            //CHECK(cudaMallocHost(&ptr, size)); // for cuda stream async
            ptr = malloc(size);
            memset(ptr, 0, size); 

            return ptr;
        }
        // 2.big buf, 先去bigblocks里面找空闲的（free出来且未归还到OS的）
        if (size > 1024 * 1024) { // > 1M
            auto &BigBlocks = cudaBigBlocksMap[dev_id]; 

            int blockID = -1; 

            for (int i = 0; i < BigBlocks.size(); i++) {
                if (BigBlocks[i].size >= size && !BigBlocks[i].is_allocated //判定条件1.池子中的size要大于需求的size 且 判定条件2.这块buffer还没被分配出去
                    && BigBlocks[i].size - size < 1 * 1024 * 1024) {        //判定条件3. 为避免内存碎片化过于严重，池中的这个size还不能比需求的size大太多。这里定义为1MB

                    if (blockID == -1 || BigBlocks[blockID].size > BigBlocks[i].size) { 
                        blockID = i;
                    }
                }
            }

            if (blockID != -1) {
                BigBlocks[blockID].is_allocated = true;
   
                return BigBlocks[blockID].data;
            }
            // // 大池中没找到空闲的，只能通过cudaMalloc分配了
            void* new_buffer;
            cudaMalloc(&new_buffer, size);
            total_allocated_size += size;

            BigBlocks.push_back(CudaBigBlock(new_buffer, size, true));//把这个new buffer加到big blockpool里面去

            return new_buffer;
        }
        // 3.small buf, 先去smallblocks里面找空闲的（free出来且未归还到OS的）
        auto &SmallBlocks = cudaSmallBlocksMap[dev_id];
        for (int i = 0; i < SmallBlocks.size(); i++) {
            if (SmallBlocks[i].size >= size && !SmallBlocks[i].is_allocated) {//判定条件1.池子中的size要大于需求的size 且 判定条件2.这块buffer还没被分配出去
                SmallBlocks[i].is_allocated = true;
                FreeSize[i] += SmallBlocks[i].size;

                return SmallBlocks[i].data; 
            }
        }
        // 小池中没找到空闲的，只能通过cudaMalloc分配了
        void* new_buffer = (void*)ptr;
        CHECK(cudaMalloc(&new_buffer, size));
        CHECK(cudaMemset(new_buffer, 0, size));  

        SmallBlocks.push_back(CudaSmallBlock(new_buffer, size, true));  

        return new_buffer; 
    }

    /////////////4. 定义unifyfree释放内存(包括1.通过free函数直接释放cpu内存 2.清理碎片 3.free一些需要清理的block(小池子里的放到freesize，大池子的直接设为flase不归还OS) 4.前三步都没找到，直接cudafree掉)////////////

    void UnifyFree(void* ptr, bool is_host) { 
        if (ptr == nullptr) {
            return;
        }
        // 1.host free 
        if (is_host) { //如果是cpu上的内直接free
            free(ptr);
            return;
        }
        // 2.清理碎片：当累计的小buf超出了1G时，清理未分配出去的smallblocks, 已分配的还是保留在smallmap
        for (auto &it: cudaSmallBlocksMap) {  
            if (FreeSize[it.first] > 1024 * 1024 * 1024) { 
                auto &cudaBlocks = it.second;  

                std::vector<CudaSmallBlock> temp;//

                for (int i = 0; i < cudaBlocks.size(); i++) {   
                    if (!cudaBlocks[i].is_allocated) { 
                        cudaSetDevice(it.first); 
                        cudaFree(cudaBlocks[i].data); 

                    } else {
                        temp.push_back(cudaBlocks[i]);
                    }
                }
                cudaBlocks.clear(); 
                it.second = temp; 
                FreeSize[it.first] = 0; 
            }
        }
        // 3.找到待free的buffer的位置，设is_allocated = false，大小block都不归还到OS，除非没有在大小block里面找到待free的ptr
        // 大块清理分配比较耗时，为了降低损耗，用标记为标记为已经清除即可，时间复杂度仅为查询map的时间复杂度O(1)
        for (auto &it: cudaSmallBlocksMap) {
            auto &cudaBlocks = it.second;
            for (int i = 0; i < cudaBlocks.size(); i++) {
                if (cudaBlocks[i].data == ptr) {  
                    FreeSize[it.first] += cudaBlocks[i].size;
                    cudaBlocks[i].is_allocated = false;
                    return;
                }
            }
            auto &bigBlocks = cudaBigBlocksMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                if (bigBlocks[i].data == ptr) {
                    bigBlocks[i].is_allocated = false;
                    return;
                }
            }
        }
        //4.如果没有找到，直接free掉
        cudaFree(ptr);    
    }
};
