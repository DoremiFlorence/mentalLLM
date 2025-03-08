import requests
import json
from req import chat_request
from score import rank
from rewrite import in_rewrite
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

# 创建问题序列
questions = [
    "您是否经常感到心情低落？",
    "您是否有睡眠困难、难以入睡或者睡眠过多？",
    "您是否经常感到疲劳或无精打采？",
    "您的胃口是否不好，或者比平常吃得更多？",
    "您是否觉得自己做得很糟，或者感到自己失败了，或让自己或家人失望？",
    "您是否发现自己难以集中注意力，如阅读报纸或看电视时？",
    "您是否感到动作或说话速度变慢到别人可以察觉，或者相反，您是否感到异常烦躁或坐立不安？",
    "您是否曾有过不如死掉或以某种方式伤害自己的念头？",
    "这些问题是否在您工作、处理家庭事务或与他人相处时造成了困难？如果有，困难的程度如何？"
]

# 用于存储已搜索过的句子
searched_sentences = []


def get_sentence_embeddings(sentences):
    # 使用模型和标记器编码句子
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取每个句子的隐藏状态
    sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return sentence_embeddings


def search_related_sentence(sentences, user_input, top_k=1):
    # 获取句子的嵌入
    sentence_embeddings = get_sentence_embeddings(sentences)

    # 用户输入的嵌入
    user_input_embedding = get_sentence_embeddings([user_input])

    # 计算用户输入与每个句子之间的余弦相似度
    similarities = cosine_similarity(user_input_embedding, sentence_embeddings)

    # 获取相似度最高的句子的索引
    top_k_indices = similarities.argsort(axis=1)[0][-top_k:][::-1]

    # 返回与用户输入最相关的句子，并将其添加到已搜索列表中
    related_sentences = [sentences[i] for i in top_k_indices if sentences[i] not in searched_sentences]
    searched_sentences.extend(related_sentences)

    return related_sentences


print("您是否觉得做任何事情都没有兴趣或乐趣？")
# 多轮问答
while True:
    user_input = input("请输入您的回答（输入 q 退出）：")
    if user_input.lower() == 'q':
        print("谢谢使用，再见！")
        break

    related_sentences = search_related_sentence(questions, user_input)

    if related_sentences:
        print("与您的回答最相关的句子是：", related_sentences[0])
    else:
        # 如果没有找到相关句子，就随机选择一个未输出的句子输出
        not_searched_sentences = [s for s in questions if s not in searched_sentences]
        if not_searched_sentences:
            random_sentence = random.choice(not_searched_sentences)
            print("未找到与您的问题相关的句子。随机选择一个句子输出：", random_sentence)
            searched_sentences.append(random_sentence)
        else:
            print("未找到与您的问题相关的句子，并且所有句子都已经输出。")

# score = []
# # 使用enumerate函数在for循环中处理索引和值
# for question in questions:
#     # 输出问题
#     print(question)
#     answer = input()
#     qa = question + answer
#     # rew = in_rewrite(qa)
#     reply = chat_request(qa)
#     # print(rank(qa))
#     print(reply)
    # score.append(int(rank(qa)))
#
# sum_score = sum(score)
# print(sum_score)
