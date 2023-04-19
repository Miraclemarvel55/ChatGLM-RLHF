import torch
def generate_inputs(tokenizer, query='', history=[]):
    assert query or history, "query and history cannot both empty"
    if not history:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            if i==len(history)-1 and query == "":
                prompt += "[Round {}]\n问：{}\n答：".format(i, old_query)
            else:
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
        if query:
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    inputs = tokenizer([prompt], return_tensors="pt")
    gen_len = 0
    if query=="":
        # query为空代表history的最后一个回答是目标答案
        last_response_encode = tokenizer.encode(history[-1][1], return_tensors="pt", add_special_tokens=False)
        if last_response_encode[0, 0] == 5:
            last_response_encode = last_response_encode[:, 1:]
            # TODO batch化
        eops = torch.zeros_like(last_response_encode[:, :1])+tokenizer.convert_tokens_to_ids("<eop>")
        # TODO 后续用scatter来放到可能多个句子的带padding的正确位置，暂时先放到最后，因为现在只有一句
        last_response_encode = torch.cat([last_response_encode, eops], dim=-1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], last_response_encode], dim=-1)
        gen_len = last_response_encode.shape[1]
    return inputs, gen_len

if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    inputs = generate_inputs(tokenizer, query="", history=[["你好", "你好"]])
    print(inputs)
    inputs2 = generate_inputs(tokenizer, query="你好", history=[["你好", "你好"]])
    print(inputs2)

