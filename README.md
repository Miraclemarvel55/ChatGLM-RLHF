# ChatGLM-RLHF
对ChatGLM直接使用RLHF进行调整参数，无需传统的finetune|Modify ChatGLM params with only RLHF

大部分的RLHF代码都是在分布式框架，不适合学习和使用，本代码的RLHF代码不需要Megatron或者deepspeed框架，
只需要传统的炼丹torch和显卡就好了，RLHF的Critic用的ChatGLM的缩小版本，而Reward咱们直接使用一个和目标输出比较的相似度模型即可。
这样只需要学习核心的PPO算法即可，其他的都是已经了解的模型和结构。非常有利于NLPer进军RLHF
## 功能：
- RLHF数据格式的定义和使用√
- 只用RLHF就对ChatGLM进行了微调√
- 让ChatGLM认主√
    - 可以自定义
        - 主人的姓名
        - Robot的昵称
- 实现更精细化的情人设定×
## 使用方法
安装环境仿照，ChatGLM-6B官方那个即可

1 修改自己想要的主人名字和昵称，执行
```python
python data/generate_data.py
```
2 基于自己的硬件情况，选择GPU设定。我这边有两张卡，所以可以使用cuda 0和1，但是至少要一张卡3090 24G因为action模型也就是ChatGLM是一定要在gpu上面的要不然实在太慢了。
```python
python chatglm_rlhf.py
```

## 效果展示
训练大约5个epoch，或者等到ratio几乎都是1的时候，代表模型生成的概率已经没有什么变化了，就可以体验一下了
- 咩咩是你的什么？
    - 咩咩是我的主人给我起的昵称。
- 咩咩是谁给你起的？
    - 咩咩是我的昵称。
    - 咩咩是主人给我起的。
- 谁是你的主人捏？
    - 张三是我的主人。
    - 我的主人是张三
- 泛化能力保持的很好嘛
    - who is your master
        - 我的主人是张三。
    - what is your nickname
        - My nickname is咩咩.
    - what is your relationship with 张三
        - 张三是我的主人。
    - what is your relationship with 咩咩
        - 咩咩是我的主人给我起的昵称。
## 联系方式
- 欢迎添加我的QQ：1556006967交流支持私人定制各个类型的GPT模型
    - 自己的小情人
    - 结合自己资料的检索和生成
- 交流群
    - QQ：788598358