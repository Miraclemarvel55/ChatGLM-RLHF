

# "说明": "使用时加载替换格式化名字，生成训练数据，自己设定各个称呼进行替换，制定自己专属的机器人助手，其中每一个list里面可能是多个问答对。",
#新的三者关系
#  开发者：原始开发预训练模型的人
#  用户也就是主人
#  Robot昵称
from pathlib import Path
import json
import os
file_dir = os.path.dirname(__file__)

profile_dict = {
        "xxx": "主人的称呼",
        "nn": "给机器人的昵称",
        "yyy": "主人的称呼负样本",
        "mm": "昵称负样本",
    }
profile_dict = {
        "xxx": "张三",
        "nn": "咩咩",
        "yyy": "李四",
        "mm": "咕咕",
    }

unformatted_path = os.path.join(file_dir, "profile.json")
data_unformated = json.loads(Path(unformatted_path).read_text())


dialogues = []
for dialogue_unformated in data_unformated:
    dialogue = []
    for ix, turn_ in enumerate(dialogue_unformated):
        turn = dict()
        for k, v in turn_.items():
            if isinstance(v, str):
                turn[k] = v.format(**profile_dict)
            elif isinstance(v, list):
                turn[k] = [s.format(**profile_dict) for s in v]
        dialogue.append(turn)
    dialogues.append(dialogue)

save_path = os.path.join(file_dir, "profile_instance.json")
Path(save_path).write_text(json.dumps(dialogues, ensure_ascii=False, indent=1))