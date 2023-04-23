import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
import inspect
import sys
print(sys.path)
import torch
from itertools import chain
import json
from pathlib import Path
from tqdm import tqdm
from pprint import pprint
from transformers import AutoTokenizer, AutoModel
from chatglm_local.modeling_chatglm import ChatGLMForConditionalGeneration
from models_rlhf import Critic, Reward, RewardBySimilarity
from utils_chatglm import generate_inputs
import random
from collections import defaultdict

# set device
action_device = "cuda:1"
RM_device = "cpu" #"cuda:0"
critic_device = "cuda:0" # "cpu" 

reward_model = RewardBySimilarity(device=RM_device)
critic = Critic(device=critic_device)
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
if "cuda" in action_device:
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = model.half().cuda(action_device) # half for gpu only
elif "cpu" == action_device:
    model = ChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).bfloat16()

# 只更新embedding
model.requires_grad_(False)
model.transformer.word_embeddings.requires_grad_(True)
# model.lm_head.requires_grad_(True)
# model.lm_head.weight和model.transformer.word_embeddings.weight是共享参数，两者是转置关系


decay_up_matrix_T = None
def get_decay_up_matrix_T(dtype=torch.float, device="cpu", max_length = 2048, gamma=0.99, tau=0.95):
    global decay_up_matrix_T
    if decay_up_matrix_T is None:
        # 生成衰减矩阵
        decay = gamma*tau
        decay_row = torch.ones(max_length, dtype=dtype, device=device)*decay
        decay_row[0] = 1
        decay_row_cross_time = decay_row.cumprod(dim=-1)
        assert decay_row_cross_time.sign().min() == 0
        decay_up_matrix = torch.zeros((max_length, max_length), dtype=dtype, device=device)
        for i in range(max_length):
            decay_row = decay_row_cross_time.roll(i)
            decay_row[:i] = 0 # 确保看不见前面的
            decay_up_matrix[i] = decay_row
        decay_up_matrix_T = decay_up_matrix.T# 先进行转置，因为后面需要用到矩阵乘法
    return decay_up_matrix_T

def gae_vectorize(values, rewards, masks=None):
    """
        values:表示各个时间步状态的状态值。shape:batch_size,sequence_length
        rewards:表示各个时间步做出的动作的奖励，对于gpt当前动作也是动作对应的下一状态。所以shape和values一样
                # 注意这里的rewards表示当前动作状态的reward
        masks:由于是要对生成的actions做gae，也就是泛化优势估计，
                # 所以类似以往的mask只需要对padding进行mask，
                # 因为padding的delta会被放入加权计算，而action前面的delta，
                # 由于生成的衰减矩阵就是上三角的，自然就看不到前面的。
                # 0表示mask， 1表示需要的。
    """
    action_rewards = rewards.roll(-1) # 当前状态的动作的奖励是下一个状态出现时给出的，而奖励是基于状态计算的，所以需要shift一个时间步回去
    # 为了学到最后输出的<eop>,所以给最后的状态赋予一个rewards试试
    action_rewards = (action_rewards+rewards)/2 # 将奖励分配到最后两步

    values_estimator_1_order = action_rewards + values.roll(-1) # 这里要注意roll是循环的，所以最后一位的值可能不能用
    deltas = values_estimator_1_order - values  #必须要action+下一个时刻的值函数减去当前值函数，这是表示当前action的优势
    # get weight matrix
    decay_up_matrix_T = get_decay_up_matrix_T(dtype=deltas.dtype, device= deltas.device)
    # 计算gae
    max_goal_length = deltas.shape[-1]
    sub_decay_up_matrix_T = decay_up_matrix_T[:max_goal_length, :max_goal_length]
    if masks is not None:
        deltas = deltas * masks
    gae = deltas.matmul(sub_decay_up_matrix_T.to(deltas.device))
    assert gae.shape == deltas.shape
    return gae

def get_log_prob(generated_outputs, input_ids, gen_method = "greedy_search"):
    # beam_search generate 给出来的scores就是log_prob了，所以直接gather获取即可
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:] 
    # let's stack the logits generated at each step to a tensor
    # 要小心greedy search 拿到的是score，需要再log_softmax
    # 而beam_search 拿到的已经是log_softmax了
    scores = torch.stack(generated_outputs.scores, dim=1)
    # if scores.max() >0 :
    #     gen_method = "greedy_search"
    if gen_method == "beam_search":
        log_prob_stacked = scores
    else:
        log_prob_stacked = torch.stack(generated_outputs.scores, dim=1).log_softmax(dim=-1)
    # now we need to collect the log_prob of the generated token # we need to add a dummy dim in the end to make gather work 
    log_prob = torch.gather(log_prob_stacked, 2, gen_sequences[:, :, None]).squeeze(-1)
    return log_prob

def get_log_probs_with_input_ids(states, gen_max_len):
    input_ids = states
    model_inputs = model.prepare_inputs_for_generation(input_ids)
    output = model(**model_inputs)  #将已经生成的序列放进去计算，再次计算得到目标action也就是后续字符的概率或者log_prob值
    logits = output.logits[:, -(gen_max_len+1):-1].log_softmax(dim=-1) # 比先softmax再log好,复杂度减小，并且解决些nan问题
    new_log_probs = logits.gather(dim=-1, index=input_ids[:, -gen_max_len:].unsqueeze(-1)).squeeze(-1)
    return new_log_probs

def sample_history_from_turns(turns):
    history = [ [turn["问"], random.choice(turn["好答"])] for turn in turns ]
    return history


optimize_params = list(model.transformer.word_embeddings.parameters())+list(critic.parameters())
from torch.optim import Adam
optimizer = Adam(optimize_params, lr=1e-4, eps=1e-3)
qa_logs = defaultdict(list)

def main(prompts_path):
    dataset = json.loads(Path(prompts_path).read_text(encoding="utf8"))
    for epoch in range(100):
        for ix, turns in enumerate(tqdm(dataset, mininterval=1)):
            history = sample_history_from_turns(turns)
            good_answers = turns[-1]["好答"]
            bad_answers = turns[-1]["坏答"]
            history_ = history
            r = random.randint(1, 5)
            if r>3:
                query = history[-1][0]
                history_ = history[:-1]
            else:
                # 将目标句直接用RL提升或降低它的概率，得到类似finetune的效果
                query = ""
            inputs, gen_len = generate_inputs(tokenizer, query=query, history=history_)
            input_ids = inputs["input_ids"].to(action_device)
            if query != "":
                num_beams, num_return_sequences = 1, 1 # 3, 2 # set bigger if you have bigger compute memory
                assert num_beams >= num_return_sequences, "candidates num should greater than returns num"
                max_new_tokens = 8
                gen_method = "greedy_search" if num_beams == 1 else "beam_search" 
                generate_ = model.generate(input_ids=input_ids, do_sample=False, num_beams=num_beams, max_new_tokens=max_new_tokens,
                                    num_return_sequences=num_return_sequences, use_cache=True, num_beam_groups=1, output_scores=True,
                                    output_hidden_states=False, return_dict_in_generate=True)
                sequences = generate_.sequences
                log_probs = get_log_prob(generated_outputs=generate_, input_ids=input_ids, gen_method=gen_method)
                gen_texts = tokenizer.batch_decode(sequences[:, input_ids.shape[1]:])
                out_texts = tokenizer.batch_decode(sequences)
                qa_logs[query].extend(gen_texts)
                print(query, qa_logs[query], sep="\n")
            else:
                # 将目标句直接用RL提升或降低它的概率，得到类似finetune的效果
                sequences = input_ids
                with torch.no_grad():
                    log_probs = get_log_probs_with_input_ids(input_ids, gen_max_len=gen_len)
                gen_texts = [history[-1][1]]
                out_texts = tokenizer.batch_decode(sequences)
                print("目标句直接用RL提升它的概率：", out_texts)

            # compute reward for generated sequences
            reward = reward_model(gen_texts=gen_texts, good_answers=good_answers, bad_answers=bad_answers).unsqueeze(1)
            assert reward.shape == (len(gen_texts), 1), "need unsqueeze for next scatter_"
            rewards = torch.zeros_like( sequences, dtype=reward.dtype, device=reward.device)
            pad_id = tokenizer.convert_tokens_to_ids("<pad>")
            masks = ( sequences!=pad_id).long().to(RM_device)
            final_position = masks.sum(dim=-1)-1
            index=final_position.unsqueeze(-1)
            rewards.scatter_(dim=1, index=index, src=reward)
            # 确保都放到values所在的device
            rewards = torch.tensor(rewards, dtype=critic.dtype, device=critic.device)
            masks = masks.to(critic.device)
            def ppo(ppo_epochs=5, states= sequences,log_probs=log_probs, rewards=rewards, masks=masks, clip_param=0.2):
                for ppo_epoch in range(ppo_epochs):
                    # compute new log probs
                    new_log_probs = get_log_probs_with_input_ids(states, log_probs.shape[1])
                    entropy = 0 # 暂时不需要熵的约束
                    # compute value
                    # 到奖励模型和值函数模型的输入可以是一样的都是生成的序列。
                    # 生成序列同时包括state和next action
                    # prepare input for critic model
                    input_ids_critic =  states.to(critic_device)
                    values = critic(input_ids=input_ids_critic)
                    # compute gae
                    gae = gae_vectorize(values=values, rewards=rewards, masks=masks)
                    advantages = gae[:, -log_probs.shape[-1]:].to(new_log_probs.device)
                    # 计算value的估计量的偏差作为actor loss
                    # 以及ppo的actor_loss
                    value_estimator_delta = advantages
                    ratio = (new_log_probs - log_probs).exp()
                    print("reward",reward, "ratio:", ratio, sep="\n")
                    if torch.isinf(ratio).any():
                        break
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
                    actor_loss  = - torch.min(surr1, surr2).mean()
                    critic_loss = value_estimator_delta.square().mean()
                    loss = 0.5 * (critic_loss + actor_loss) - 0.001 * entropy
                    # optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print("loss", loss)
            torch.cuda.empty_cache()
            ppo()

if __name__ == "__main__":
    file_dir = os.path.dirname(__file__)
    dialogues_path = os.path.join(file_dir, "data", "profile_instance.json")
    main(prompts_path = dialogues_path)

