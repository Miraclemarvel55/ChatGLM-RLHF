# ChatGLM-RLHF
对ChatGLM直接使用RLHF进行调整参数，无需传统的finetune|Modify ChatGLM params with only RLHF
## 功能：
- RLHF数据格式的定义和使用√
- 只用RLHF就对ChatGLM进行了微调√
- 让ChatGLM认主√
    - 可以自定义
        - 主人的姓名
        - Robot的昵称
- 实现更精细化的情人设定×
## 使用方法
1 修改自己想要的主人名字和昵称，执行
```python
python data/generate_data.py
```
2 基于自己的硬件情况，选择GPU设定。我这边有两张卡，所以可以使用cuda 0和1，但是至少要一张卡3090 24G因为action模型也就是ChatGLM是一定要在gpu上面的要不然实在太慢了。

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