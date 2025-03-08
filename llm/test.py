import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
import random

# 设置模型路径
MODEL_PATH = os.environ.get('MODEL_PATH', '/home/zhengjinfang/bge-large-zh-v1.5')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# 加载模型和标记器
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval()

# 10句话的列表，按类别分组
sentences_by_category = {
    # 情绪状态
    "category_1": [
        "您是否经常因为一些小事而烦恼？",
        "即使家属，朋友想帮你，您是否仍然无法摆脱心中的苦闷？",
        "您是否经常觉得意志消沉？",
        "您是否经常感到害怕？",
        "您是否经常感到高兴？",
        "您是否经常会哭泣？",
        "您是否经常感到忧愁？",
        "您是否经常觉得人们不喜欢您？",
        "您是否经常感到情绪沮丧，郁闷？",
        "您是否经常在早晨心情最好？",
        "您是否经常想哭或感到需要哭泣？",
        "您是否经常无故感到疲劳？",
        "您是否经常比平时更容易激怒？",
        "您是否经常做什么事都感到没有兴趣或乐趣？",
        "您是否经常感到心情低落？",
        "您是否经常感到疲劳或无精打采？",
        "您是否经常有不如死掉或用某种方式伤害自己的念头？"
    ],

    # 生理症状
    "category_2": [
        "您是否经常不太想吃东西，胃口不好？",
        "您是否经常睡眠情况不好？",
        "您是否经常夜间睡眠不好？",
        "您是否经常吃饭像平常一样多？",
        "您是否经常感到体重减轻？",
        "您是否经常为便秘烦恼？",
        "您是否经常感到心跳比平时快？",
        "您是否经常睡眠困难，很难熟睡或者睡太多？",
        "您是否经常胃口不好或吃太多？"
    ],
    # 认知功能
    "category_3": [
        "您是否经常觉得自己的生活是失败的？",
        "您是否经常觉得生活得很有意思？",
        "您是否经常觉得前途是有希望的？",
        "您是否经常觉得和一般人一样好？",
        "您是否经常对未来感到有希望？",
        "您是否经常觉得决定什么事情很容易？",
        "您是否经常仍旧喜爱自己平时喜爱的东西？",
        "您是否经常感到自己是有用的和不可缺少的人？",
        "您是否经常觉得自己的生活很有意义？",
        "您是否经常觉得自己很糟，或很失败，或让自己或家人失望？"
    ],
    # 社交关系
    "category_4": [
        "您是否有时说话比平时要少？",
        "您是否经常感到孤单？",
        "您是否经常觉得别人不友善？",
        "您是否经常觉得人们不喜欢您？",
        "您是否经常坐卧不安，难以保持平静？",
        "您是否经常觉得假若我死了别人会过得更好？"
    ],
    # 日常功能
    "category_5": [
        "您是否经常觉得无法继续您的日常工作？",
        "您是否经常在做事时无法集中自己的注意力？",
        "您的头脑是否经常像往常一样清晰？",
        "您是否经常做事情像平时一样不感到困难？",
        "您是否经常注意很难集中，例如阅读报纸或看电视？",
        "您是否经常动作或说话速度缓慢到别人可察觉的程度，或者正好相反—您烦躁或坐立不安，动来动去的情况比平常更严重？",
        "以上这些问题在您工作、处理家庭事务，或与他人相处上经常造成多大的困难？"
    ]
}
all_sentence = sentences_by_category["category_1"] + sentences_by_category["category_2"] + sentences_by_category[
    "category_3"] +sentences_by_category["category_4"] + sentences_by_category["category_5"]

# 用于存储已输出过的句子
outputted_sentences = {
    "category_1": [],
    "category_2": [],
    "category_3": [],
    "category_4": [],
    "category_5": []
}


def get_sentence_embeddings(sentences):
    # 使用模型和标记器编码句子
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取每个句子的隐藏状态
    sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return sentence_embeddings


def search_related_sentence(user_input, top_k=1):
    # 遍历每个类别
    for category, category_sentences in sentences_by_category.items():
        # 如果该类别的句子还没有输出完毕
        if len(outputted_sentences[category]) < 3:
            # 获取该类别句子的嵌入
            category_embeddings = get_sentence_embeddings(category_sentences)

            # 用户输入的嵌入
            user_input_embedding = get_sentence_embeddings([user_input])

            # 计算用户输入与该类别每个句子之间的余弦相似度
            similarities = cosine_similarity(user_input_embedding, category_embeddings)

            # 获取相似度最高的句子的索引
            top_k_indices = similarities.argsort(axis=1)[0][-top_k:][::-1]

            # 返回与用户输入最相关的句子，并将其添加到已输出列表中
            related_sentences = [category_sentences[i] for i in top_k_indices if
                                 category_sentences[i] not in outputted_sentences[category]]
            outputted_sentences[category].extend(related_sentences)

            # 如果找到了相关句子，则返回
            if related_sentences:
                return "相关问题：" + related_sentences[0]

    # 如果没有找到相关句子，则随机选择一个其他类别的句子输出
    other_categories = [category for category in sentences_by_category.keys() if len(outputted_sentences[category]) < 3]
    if other_categories:
        random_category = random.choice(other_categories)
        random_sentence = random.choice(sentences_by_category[random_category])
        outputted_sentences[random_category].append(random_sentence)
        return "随机问题：" + random_sentence
    else:
        return 0


print("您是否觉得做任何事情都没有兴趣或乐趣？")
now_ques = "您是否觉得做任何事情都没有兴趣或乐趣？"
# 多轮问答
while True:
    user_input = input("请输入您的问题：")
    model_input = now_ques + user_input

    related_sentence = search_related_sentence(user_input)
    if related_sentence:
        print(related_sentence)
    else:
        print("问卷结束")
        break
