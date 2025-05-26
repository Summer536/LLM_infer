///////////////////////////////////用户主要通过这个文件的接口，把他们的输入喂到模型中///////////////////////////////
///////////////////////////////该文件接收用户的输入后，又把这些输入喂到模型里面去///////////////////////////////////
#include <stdio.h>
#include "src/utils/model_utils.h"

struct ConvertedModel { 
    std::string model_path = "/home/llamaweight/"; // 模型文件路径
    std::string tokenizer_path = "/home/llama2-7b-tokenizer.bin"; // tokenizer文件路径 （目前我们tokenizer的Encode编码还有些bug，但是我们的译码Decode是没有问题的！）
};

int main(int argc, char **argv) {
    int round = 0; 
    std::string history = ""; 

    ConvertedModel model;

    auto llm_model = llm::CreateRealLLMModel<float>(model.model_path, model.tokenizer_path);//load real weight + load tokenizer

    std::string model_name = llm_model->model_name;

    //////////////////////////////接收用户输入，这里写了一个循环是因为用户可能不只输入一次/////////////////////////////////////
    while (true) {
        printf("please input the question: ");
        std::string input;
        std::getline(std::cin, input); 

        if (input == "s") {
            break;
        }    
        

        std::string retString = llm_model->Response(llm_model->MakeInput(history, round, input), [model_name](int index, const char* content) { //调用Response函数调用我们定义的llama类（这里面处理所有的模型工作Lesson28）
            if (index == 0) {
                printf(":%s", content);
                fflush(stdout); 
            }
            if (index > 0) {
                printf("%s", content);
                fflush(stdout);
            }
            if (index == -1) {
                printf("\n");
            }
        });

        // 多轮对话保留history，和当前轮次input制作成新的上下文context
        history = llm_model->MakeHistory(history, round, input, retString); //MakeHistory是(/src/models/llama/llama.cpp)中定义好的，将历史信息和当前输出信息做一个拼接
        round++;
    }
    return 0;
}
