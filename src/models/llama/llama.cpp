#include "src/models/llama/llama.h"

// we only support batch size = 1 now
// 目前暂时只支持输入"Hey, are you conscious? Can you talk to me?"，支持了dynamic input shape后方可支持其它输入
// C++ tokenizer Encode暂时不能正常运行，正在fix，故以上输入暂时只能手动通过HF tokenizer API获取，见src/tools/HF_llama_run_script.py
// cpu unpinned buffer

////////////////////////////////////////////////////////////////////////////分配cpu buffer//////////////////////////////////////////////////////////////////////////
template <typename T>
void Llama<T>::allocateCPUBuffer(int batch_size)
{
    h_input_ids_buf_ =
        allocator->Malloc(h_input_ids_buf_, sizeof(int) * 13, true); //为什么分配了13呢？因为下方输入的prompt是13个TokenID：{1, 18637, 29892,526,366,19861, 29973,1815,366,5193,304,592,29973}
    h_input_length_buf_ =
        allocator->Malloc(h_input_length_buf_, sizeof(int) * batch_size, true);
    h_history_length_buf_ =
        allocator->Malloc(h_history_length_buf_, sizeof(int) * batch_size, true);
    h_context_length_buf_ =
        allocator->Malloc(h_context_length_buf_, sizeof(int) * batch_size, true);
    h_sequence_lengths_ =
        allocator->Malloc(h_sequence_lengths_, sizeof(int) * batch_size, true);
    h_finished_buf_ = allocator->Malloc(h_finished_buf_, sizeof(bool) * batch_size, true);
    for (int i = 0; i < batch_size; i++)
    {
        h_finished_buf_[i] = 0;
    }
    h_output_ids = allocator->Malloc(h_output_ids, sizeof(int) * batch_size, true);
}


////////////////////////////////////////////////////////////////////////////分配GPU buffer//////////////////////////////////////////////////////////////////////////
// alloc gpu buffer
template <typename T>
void Llama<T>::allocateGPUBuffer(int batch_size)
{
    step = new TensorWrapper<int>(CPU, getTensorType<int>(), {1});
    layer = new TensorWrapper<int>(CPU, getTensorType<int>(), {1}, &layer_id);
    // for context decoder
    context_decoder_input = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*token num*/ 13, hidden_units});
    context_decoder_output = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*token num*/ 13, hidden_units});
    // split from context_decoder_output
    context_decoder_lmhead_input = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*token num*/ 1, hidden_units});
    // for self decoder
    decoder_input = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*batch size*/ 1, hidden_units});
    decoder_output = new TensorWrapper<T>(GPU, getTensorType<T>(), {/*batch size*/ 1, hidden_units});
    input_ids = new TensorWrapper<int>(GPU, getTensorType<int>(), {/*token num*/ 13}); 
    // for context decoder
    input_length = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    history_length = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    context_length = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    sequence_lengths = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    // kv cache buffer
    all_k_cache = new TensorWrapper<T>(GPU, getTensorType<T>(), {num_layers, batch_size, kv_head_num, max_seq_len, head_size});  //kvcache的设置，这里其实设置为max_seq_len比较朴素，好多序列其实没有这么大的长度
    all_v_cache = new TensorWrapper<T>(GPU, getTensorType<T>(), {num_layers, batch_size, kv_head_num, max_seq_len, head_size});
    token_ids = new TensorWrapper<int>(GPU, getTensorType<int>(), {batch_size});
    is_finished = new TensorWrapper<bool>(GPU, getTensorType<bool>(), {batch_size});
    output_rmsnorm_weight = new TensorWrapper<T>(GPU, getTensorType<T>(), {hidden_units}, llama_weights->out_rmsnorm_weight.gamma);
    probs = new TensorWrapper<T>(GPU, getTensorType<T>(), {batch_size, vocab_size});
    unused_residual = new TensorWrapper<T>(GPU, getTensorType<T>(), {batch_size, hidden_units});
    // allocate buffer of above
    unused_residual->data = allocator->Malloc(unused_residual->data, sizeof(T) * 13 * hidden_units, false);;
    context_decoder_input->data =
        allocator->Malloc(context_decoder_input->data, sizeof(T) * 13 * hidden_units, false);
    context_decoder_output->data =
        allocator->Malloc(context_decoder_output->data, sizeof(T) * 13 * hidden_units, false);
    context_decoder_lmhead_input->data =
        allocator->Malloc(context_decoder_lmhead_input->data, sizeof(T) * 1 * hidden_units, false);
    decoder_input->data = allocator->Malloc(decoder_input->data, sizeof(T) * batch_size * hidden_units, false);
    decoder_output->data = allocator->Malloc(decoder_output->data, sizeof(T) * batch_size * hidden_units, false);

    input_ids->data = allocator->Malloc(input_ids->data, sizeof(int) * 13, false);
    input_length->data = allocator->Malloc(input_length->data, sizeof(int) * batch_size, false);
    history_length->data = allocator->Malloc(history_length->data, sizeof(int) * batch_size, false);
    context_length->data = allocator->Malloc(context_length->data, sizeof(int) * batch_size, false);
    sequence_lengths->data = allocator->Malloc(sequence_lengths->data, sizeof(int) * batch_size, false);

    all_k_cache->data = allocator->Malloc(all_k_cache->data, sizeof(T) * num_layers * batch_size * max_seq_len * kv_head_num * head_size, false);
    all_v_cache->data = allocator->Malloc(all_v_cache->data, sizeof(T) * num_layers * batch_size * max_seq_len * kv_head_num * head_size, false);

    token_ids->data = allocator->Malloc(token_ids->data, sizeof(int) * batch_size, false); // 4,,进位到32x为32

    is_finished->data = allocator->Malloc(is_finished->data, sizeof(bool) * batch_size, false); // 4,,进位到32x为32
    probs->data = allocator->Malloc(probs->data, sizeof(T) * batch_size * vocab_size, false);
    // topK buffer
    topk_id = new TensorWrapper<int>(GPU, getTensorType<int>(),
                                     {batch_size, beamwidth, BlockPerBeam, K});
    topk_val = new TensorWrapper<T>(GPU, getTensorType<T>(), {batch_size, beamwidth, BlockPerBeam, K});
    final_topk_id = new TensorWrapper<int>(GPU, getTensorType<int>(),
                                           {batch_size * beamwidth, K});
    final_topk_val = new TensorWrapper<T>(GPU, getTensorType<T>(), {batch_size * beamwidth, K});
    topk_id->data = allocator->Malloc(topk_id->data, sizeof(int) * batch_size * beamwidth * BlockPerBeam * K, false);
    topk_val->data = allocator->Malloc(topk_val->data, sizeof(T) * batch_size * beamwidth * BlockPerBeam * K, false);
    final_topk_id->data = allocator->Malloc(final_topk_id->data, sizeof(int) * batch_size * beamwidth * K, false);
    final_topk_val->data = allocator->Malloc(final_topk_val->data, sizeof(T) * batch_size * beamwidth * K, false);
}


////////////////////////////////////////////////////////////////////////////free buffer//////////////////////////////////////////////////////////////////////////
// free CPU and GPU buffer
template <typename T>
void Llama<T>::free()
{
    allocator->Free(h_input_ids_buf_, true);
    allocator->Free(h_input_length_buf_, true);
    allocator->Free(h_history_length_buf_, true);
    allocator->Free(h_context_length_buf_, true);
    allocator->Free(h_sequence_lengths_, true);
    DeviceSyncAndCheckCudaError();
    allocator->Free(context_decoder_input->data);
    allocator->Free(context_decoder_output->data);
    allocator->Free(decoder_input->data);
    allocator->Free(decoder_output->data);
    allocator->Free(input_ids->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(input_length->data);
    allocator->Free(history_length->data);
    allocator->Free(context_length->data);
    allocator->Free(sequence_lengths->data);
    allocator->Free(all_k_cache->data);
    allocator->Free(all_v_cache->data);
    allocator->Free(token_ids->data);
    allocator->Free(is_finished->data);
    allocator->Free(probs->data);
    DeviceSyncAndCheckCudaError();
}

///////////////////////////////////InitializeForContextDecoder函数：初始化Ctxdecoder///////////////////////////////////////////
template <typename T>
void Llama<T>::InitializeForContextDecoder(IntDict &int_params_first_token)
{
    // only support and assumed bs = 1
    h_input_length_buf_[0] = int_params_first_token["cur_input_length"]; //将字典里面的参数赋值到cpu的buffer上，这些cpubuffer是在allocateCPUBuffer函数中赋值的！
    h_history_length_buf_[0] = int_params_first_token["history_length"];
    h_context_length_buf_[0] = int_params_first_token["context_length"];
    CHECK(cudaMemcpy(input_ids->data,                                   //然后将cpu上的值搬运到GPU上
                     h_input_ids_buf_,  // get from tokenizer encode
                     sizeof(int) * h_input_length_buf_[0],
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(input_length->data, h_input_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice)); //然后将cpu上的值搬运到GPU上
    CHECK(cudaMemcpy(history_length->data, h_history_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(context_length->data, h_context_length_buf_, sizeof(int) * batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(is_finished->data, h_finished_buf_, sizeof(bool) * batch_size, cudaMemcpyHostToDevice));
}

///////////////////////////////////InitializeForSelfDecoder函数：初始化Selfdecoder///////////////////////////////////////////
template <typename T>
void Llama<T>::InitializeForSelfDecoder()
{
    // nothing to do now  //暂时没啥可以初始化的
}


//note: 返回所有轮次总共的input、总共input中的history部分、总共input中的当前轮次input部分
template <typename T>
std::vector<std::string> Llama<T>::MakeInput(const std::string &history, int round, const std::string &input) ///考虑到多轮对话，这里将用户当前的input于之前的history合并一下，作为prompt输入给模型
{
    std::vector<std::string> ret = {(round == 0 ? "" : history) + input, history, input};
    return ret;
}


template <typename T>
//note: 根据第round轮的结果制作history
std::string Llama<T>::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output)
{
    return (round == 0 ? prompt : history) + input + output; 
}


//note: input embedding kernel wrapper
template <typename T>
void Llama<T>::inputEmbedding(TensorWrapper<int> *input_ids, TensorWrapper<T> *decoder_input)
{
    launchInputEmbedding<T>(input_ids, decoder_input, &(llama_weights->pre_decoder_embedding_weight));
    DeviceSyncAndCheckCudaError();
}


//////////////////////////////////////////////////////////////////////firstTokenGen函数：生成每轮对话的第一个Token(context decoder)/////////////////////////////////////////////////////////////////
////////////////////包括了InitializeCtxdecoder + Ctxdecoder的forward + RMSNorm + LMHeadAndTopKSample --> Generate 1st tokenid//////////////////
//  note: 每轮对话的1st token generation / context decoder
template <typename T>
int Llama<T>::firstTokenGen(LLaMAAttentionDynParams &dparams, IntDict &int_params_first_token)
{   
    InitializeForContextDecoder(int_params_first_token);
    //1.embedding操作
    inputEmbedding(input_ids, context_decoder_input);

    LLM_CHECK_WITH_INFO(context_decoder_input->data != nullptr, "GPU context decoder input data is not initialized");
    LLM_CHECK_WITH_INFO(history_length->data != nullptr, "GPU history_length data is not initialized");
    LLM_CHECK_WITH_INFO(input_length->data != nullptr, "GPU input_length data is not initialized");
    LLM_CHECK_WITH_INFO(context_length->data != nullptr, "GPU context_length data is not initialized");
    LLM_CHECK_WITH_INFO(output_rmsnorm_weight->data != nullptr, "GPU output_rmsnorm_weight data is not initialized");

    TensorMap decoder_inputs{
        {"decoder_input", context_decoder_input},
        {"history_length", history_length},
        {"input_length", input_length},
        {"context_length", context_length},
        {"output_norm_weight", output_rmsnorm_weight}, //就是Ctxdecoder中的最后一步RMSNorm的时候需要用到的weights // located at llamaweights class, rather not llamalayerweigths
        {"layer_id", layer}};
    // output buffer and input buffer are shared to reuse buffer between layers
    // I dont rewrite Tensor's copy constructor, default shallow copy, that can share buffer, which is I want
    TensorMap decoder_outputs{
        {"decoder_output", context_decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}};
    
    //2.Ctxdecoder的forward函数
    context_decoder->forward(decoder_inputs,
                             llama_weights->llama_layer_weight, // layerWeights包含着llama_layer_weight,并且是public的
                             decoder_outputs,
                             dparams);

    //3.Ctxdecoder的最后一个RMSNorm函数
    // output rmsnorm
    Tensor* decoder_output = decoder_outputs["decoder_output"];
    launchRMSNorm(decoder_output->as<T>(), //in&out, [bs, q_hidden_units]
                  unused_residual, //残差
                  llama_weights->out_rmsnorm_weight,//rmsnorm weights, [q_hidden_units]    //layerWeights同样包含着out_rmsnorm_weight,并且是public的
                  rmsnorm_eps,
                  true);

    save_tensor(decoder_output->as<T>() ,"decoder_norm_out.bin");
    DeviceSyncAndCheckCudaError();

    //4.做Sampling处理
    int res = LMHeadAndTopKSample(decoder_outputs); //这个函数return h_output_ids[0]
    //std::cout << "context decoder generated  index  is " << res << "\n";
    return res;
}

//////////////////////////////////////////////////////////////////////continueTokenGen函数：生成每轮对话的第二个以及后续Token(self decoder)/////////////////////////////////////////////////////////////////
////////////////////包括了InitializeForSelfDecoder + Selfdecoder的forward + RMSNorm + LMHeadAndTopKSample --> Generate 2st and next tokenid//////////////////
template <typename T>
int Llama<T>::continueTokenGen(LLaMAAttentionDynParams &dparams)
{   
    InitializeForSelfDecoder();

    //1.embedding操作
    inputEmbedding(input_ids, decoder_input);
    TensorMap decoder_inputs{
        {"decoder_input", decoder_input},
        {"step", step}, // a batch shared same step, locate on CPU, no need GPU
        {"finished", is_finished},
        {"layer_id", layer},
        {"output_norm_weight", output_rmsnorm_weight} // located at llamaweights class, rather not llamalayerweigths
    };
    // note: 最开始是context decoder里面RoPE输出的k和v写到kv cache
    // note: self decoder之后每一个step都会输出kv到kv cache, 需要保证kv cache是llama class的成员, 这样就可以保证同步更新
    TensorMap decoder_outputs{
        {"decoder_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}};

    //2.Selfdecoder的forward函数
    self_decoder->forward(decoder_inputs,
                          llama_weights->llama_layer_weight,
                          decoder_outputs,
                          dparams);

    //3.Selfdecoder的最后一个RMSNorm函数
    // output rmsnorm
    Tensor* decoder_output = decoder_outputs["decoder_output"];
    launchRMSNorm(decoder_output->as<T>(), //in&out, [bs, q_hidden_units]
                  unused_residual,
                  llama_weights->out_rmsnorm_weight,//rmsnorm weights, [q_hidden_units]
                  rmsnorm_eps,
		  true);
    DeviceSyncAndCheckCudaError();

    //4.做Sampling处理
    int res = LMHeadAndTopKSample(decoder_outputs);
    return res;
}

//////////////////////////////////////////////////////////////////////LMHeadAndTopKSample函数：LMheadlinear和采样输出/////////////////////////////////////////////////////////////////
template <typename T>
int Llama<T>::LMHeadAndTopKSample(TensorMap &decoder_outputs)
{
    Tensor *decoder_output = decoder_outputs["decoder_output"]; //拿到first/self decoder的输出

    //1.LMhead(linear)操作
    if (index == 0) 
    {
        TensorWrapper<T> *decoder_output_tensorwrapper = decoder_output->as<T>(); 
        auto input_length = decoder_output_tensorwrapper->shape[0]; 
        auto hidden_units = decoder_output_tensorwrapper->shape[1]; 
        // fetch last token to handle ctxdecoder sampling
        auto ptr = decoder_output_tensorwrapper->data + (input_length - 1) * hidden_units;
        context_decoder_lmhead_input->data = ptr; 

        launchLinearGemm(/*Tensor**/ context_decoder_lmhead_input,                     //[1, hidden units] for ctx decoder
                         /*BaseWeight&*/ llama_weights->post_decoder_embedding_weight, // lm_head.weight.bin, [vocab_size, hidden_units] 
                         /*Tensor**/ probs,                                            //[1, vocab size] for context decoder
                         cublas_wrapper,
                         false,
                         true);

        DeviceSyncAndCheckCudaError();

    } else {    //如果是self decoder，我们用下面这段.与上方不同的点它的input/output.shape=[1, hidden units]。它直接将输入的值传入LMhead运行即可！无需再取最后一个
        launchLinearGemm(/*Tensor**/ decoder_output->as<T>(),                          //[bs, hidden_units] for self decoder
                         /*BaseWeight&*/ llama_weights->post_decoder_embedding_weight, // lm_head.weight.bin, [vocab_size,hidden_units]
                         /*Tensor**/ probs,                                            //[bs, vocab size] for self decoder
                         cublas_wrapper,
                         false,
                         true);

        DeviceSyncAndCheckCudaError();
    }

    //2.TopK
    launchTopKforBeamSearch(probs, // [bs, vocab_size]
                            topk_id,
                            topk_val,
                            final_topk_id,
                            final_topk_val); // output，这个属于是中间buffer，定义在allocatebuffer就行
    DeviceSyncAndCheckCudaError();

    int_params_of_sample.insert({"step", *step->data});//更新一下step(因为每一次生成结束后step都要加一)

    //3.Sampling
    launchSampling(/*Tensor**/ final_topk_id,          // in
                   /*Tensor**/ final_topk_val,         // in
                   /*Tensor**/ sequence_lengths,       // out, +1
                   /*Tensor**/ is_finished,            // out, 判断一下是否结束
                   /*Tensor**/ token_ids,              // out, 新生成的token ids
                   /*IntDict&*/ int_params_of_sample); // in, including step vocabsize endid
    DeviceSyncAndCheckCudaError();

    CHECK(cudaMemcpy(h_output_ids, token_ids->data, sizeof(int) * batch_size, cudaMemcpyDeviceToHost));//将采样的结果copy到cpu上，然后将这个cpu上得到的值返回
    // std::cout << "sampling d" << std::endl;
    //std::cout << "generated index: " << h_output_ids[0] << std::endl;
    return h_output_ids[0]; // only for bs = 1        
}

////////////////////////////////////////////////////////////////////////////Response函数：响应用户的输入//////////////////////////////////////////////////////////////////////////
////////////////////包括了firstTokenGen + continueTokenGen//////////////////

// 单轮对话, batch size = 1
// 返回所有轮次总共的input、总共input中的history部分、总共input中的当前轮次input部分
template <typename T>
//返回类型是一个std::string ，即每做一个推理都会返回一个TokenID，然后这个ID经过解码之后就会返回给用户
//input即用户输入的句子，我们把它重新组织了一下组成一个vector；     CallBack PrintRes：是一个回调函数。它的作用是打印模型推理出来的单词结果(位于/src/models/basemodel.h)
std::string Llama<T>::Response(const std::vector<std::string> &input, CallBack PrintRes)
{   ////////////////////////////////////因为这个bug的存在，我们当前只能做一个单轮次对话！多轮对话得等这个tokenizer.Encode修复完成后才能做！！！！！！！！！！！！！////////////////////////////////////////////////////////////////////
    //std::vector<int> res = tokenizer.Encode(input[2]); //本来是应该通过Tokenizer编码用户输入的句子的，但是目前存在一些bug，toeknzier出来的ID序列和huggingface对不上，因此我们暂时用一个硬编码替代一下（后期可以尝试优化一下这个bug！）
        // from transformers import AutoTokenizer
        // tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer_folder")
        // prompt = "Hey, are you conscious? Can you talk to me?"
        // input_ids = tokenizer(prompt, return_tensors="pt")
        // 下行的input token ids暂时是通过以上4行huggingface python api而得，修复了tokenzier.Encode之后再用以上第5行替换    std::vector<int> res = {1, 18637, 29892,526,366,19861, 29973,1815,366,5193,304,592,29973};
    std::vector<int> res = {1, 18637, 29892,526,366,19861, 29973,1815,366,5193,304,592,29973}; //这里我们暂时用一个硬编码替代一下（后期可以尝试优化一下这个bug！）

    std::string history_str = input[1]; //从input(这个input是对当前输入和之前的history做了拼接的！)中获取一下history部分，拿到一个string
    std::vector<int> history_input_ids; 
    if (!history_str.empty())
    {
        history_input_ids = tokenizer.Encode(history_str); //如果history确实有，我们就对它进行编码。    这个其实目前还不能用，因为tokenizer.Encode存在bug！（与上方一样的bug，后期可以尝试优化一下这个bug！）
    }

    // h_input_ids_buf_ = res.data();// warning: dont use this method, should use for travese assign, or the former will generate trash val out of vector's scope

    std::string total_str = input[0]; //从input中获取一下当前轮次的input部分，拿到一个string
    std::vector<int> context_ids; //声明一下context_ids
    if (!total_str.empty())
    {
        //context_ids = tokenizer.Encode(total_str);
        context_ids = {1, 18637, 29892,526,366,19861, 29973,1815,366,5193,304,592,29973}; //与上面一样，因为tokenizer.Encode存在bug，没办法只能用一个硬编码替代一下（后期可以尝试优化一下这个bug！）
    }

    //代码到现在为止，我们就可以拿到了当前输入的tokenID ：std::vector<int> res = {1, 18637, 29892,526,366,19861, 29973,1815,366,5193,304,592,29973};
    //                            以及整个历史的上下文ID： context_ids = {1, 18637, 29892,526,366,19861, 29973,1815,366,5193,304,592,29973};

    for (int i = 0; i < res.size(); i++)
    {
        h_input_ids_buf_[i] = res[i];//直接赋值移到cpu buffer上
    }
 
    int ret; 
    int context_length = context_ids.size();
    int history_length = history_input_ids.size();
    int cur_input_length = res.size();
    IntDict int_params_first_token; //IntDict是一个Map，key是一个字符串，value是一个int(位于/src/utils/params.h中)
    int_params_first_token["context_length"] = context_length;
    int_params_first_token["history_length"] = history_length;
    int_params_first_token["cur_input_length"] = cur_input_length;

    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.batch_size = 1; //目前只支持bs为1的推理
    attn_dyn_params.num_tokens = cur_input_length; //num tokens自然是指当前输入句子的长度
    attn_dyn_params.max_q_len = attn_dyn_params.num_tokens; // 这个指一个batch中的query的最大长度，因为此时不支持batch，所以就等于cur input len当前输入句子的长度
    attn_dyn_params.max_k_len = context_length;             // 这个指max context length, 指当前batch的动态最大上下文长度。还是由于此时暂时只支持一个句子的推理，因此直接赋值为context_length
    step->data = &context_length;                           // step最初为context length, 在self decoder的阶段才会用到
    //这个step已经在allocateGPUBuffer中分配好了buffer，用step表示当前生成句子的总长度，每生成一个新的token step都要加一！在self decoder阶段这个step就取代了seqlen的维度


    // retString为当前轮次对话的所有token string
    std::string retString = ""; //初始化一个字符串，它保存模型推理后得出来的结果
    //做循环推理，最多生成20个(自己定义的)英文单词
    while (index < 20) // self define output_token_limit
    {
        // kv cache here input is empty, only buffer, output is not empty
        if (index == 0) //这个index已经在头文件中初始化为0了
        {
            ret = firstTokenGen(attn_dyn_params, int_params_first_token);   //如果是第一步，则启用context decoder！
        }
        else
        {
            ret = continueTokenGen(attn_dyn_params);                        //如果是第二步及以上，则启用self decoder！
            if (ret == eos_token_id)
            {
                break;                                                      //如果到了终止TokenID，则直接返回！
            }
        }
        *(step->data) = *(step->data) + 1;//做完这个step之后要给step+1
        // std::cout << "generated index: " << ret << "\n";

        // results.push_back(ret);
        std::string genString = tokenizer.Decode({ret}).c_str(); //对输出的整型值ID做一个译码，decoder成英文单词
        retString += genString; //将多个输出拼接起来
        PrintRes(index, genString.c_str()); //回调函数(接收当前step以及当前step生成的输出(注意这里仅是当前的输出，而不是拼接好的多个输出))，实时的打印出来这个输出（在终端可以看到Token被一个个吐出来了）
        // (位于/src/models/basemodel.h)

        /////////打印输出完本step的推理结果后，我们还需要做一件事：由于在self decoder中当前step的输出将会作为下一step的输入。因此我们需要做的是在index不等于0时，将当前的step输出结果包装成InputID输入到下一step中/////
        // deep copy
        // for ctx decoder, input_ids.shape = [max_context_token_nums]
        // for self decoder, input_ids.shape = [1]
        // but  input_ids->data.size = [max_context_token_nums]
        // input_ids->shape = {1};
        if (index == 0)
        {
            TensorWrapper<int> tmp = TensorWrapper<int>(CPU, getTensorType<int>(), {1}, &ret); //将该step的输出结果 ret 包装成一个TensorWrapper
            LLM_CHECK(tmp.shape != input_ids->shape); //尽可能地做一些check，不然出了bug不太好发现。 比如这个shape应该为1，不可能等于inputid的维度13
            LLM_CHECK(tmp.dtype == input_ids->dtype);
            LLM_CHECK(tmp.location != input_ids->location);
            //将context decoder的输出的第一个Token作为下一轮的输入InputID
            allocator->Free(input_ids->data);//首先先将原先的InputID清理掉
            input_ids->data = allocator->Malloc(input_ids->data, sizeof(int) * 1, false);//在GPU上给它重新赋值
            input_ids->shape = {1};//shape也要变一下
	    CHECK(cudaMemcpy(input_ids->data, tmp.data, sizeof(int) * 1, cudaMemcpyHostToDevice)); //将cpu上的tmp.data拷贝到GPU上
        }
        else
        {   //这边已经是在selfdecoder的阶段了，直接重新拷贝即可，没有必要做context decoder那么复杂的操作了！
            CHECK(cudaMemcpy(input_ids->data, &ret, sizeof(int) * 1, cudaMemcpyHostToDevice));
        }

        index++; //更新index+1
    }
    PrintRes(-1, retString.c_str());
    return retString;
}

template class Llama<float>;
template class Llama<half>;
