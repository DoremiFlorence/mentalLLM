# -*- coding: utf-8 -*-
# @Time : 5/17/24 6:02 PM
# @Author : Florence
# @File : main.py
# @Project : llm

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
from req import chat_request
from match import implicit
from score import rank
from prompt import get_prompt
import random

# 设置模型路径
MODEL_PATH = os.environ.get('MODEL_PATH', '/home/zhengjinfang/bge-large-zh-v1.5')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# 加载模型和标记器
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval()

# 问题的列表，按类别分组
sentences_by_category = [
    # 情绪状态
    # {"text": "您是否经常因为一些小事而烦恼？", "category": "category_1"},
    {"text": "即使家属，朋友想帮你，您是否仍然无法摆脱心中的苦闷？", "category": "category_1",
     "keyword": "即使家属，朋友想帮我，仍然无法摆脱心中的苦闷"},
    {"text": "您是否经常觉得意志消沉？", "category": "category_1", "keyword": "意志消沉"},
    {"text": "您是否经常感到害怕？", "category": "category_1", "keyword": "经常害怕"},
    {"text": "您是否经常感到高兴？", "category": "category_1", "keyword": "经常高兴"},
    {"text": "您是否经常会哭泣？", "category": "category_1", "keyword": "经常哭泣"},
    {"text": "您是否经常感到忧愁？", "category": "category_1", "keyword": "经常感到忧愁"},
    {"text": "您是否经常觉得人们不喜欢您？", "category": "category_1", "keyword": "人们不喜欢我"},
    {"text": "您是否经常感到情绪沮丧，郁闷？", "category": "category_1", "keyword": "情绪沮丧"},
    {"text": "您是否经常在早晨心情最好？", "category": "category_1", "keyword": "早晨心情好"},
    {"text": "您是否经常想哭或感到需要哭泣？", "category": "category_1", "keyword": "经常想哭"},
    {"text": "您是否经常无故感到疲劳？", "category": "category_1", "keyword": "无故感到疲劳"},
    {"text": "您是否经常比平时更容易激怒？", "category": "category_1", "keyword": "比平时易怒"},
    {"text": "您是否经常做什么事都感到没有兴趣或乐趣？", "category": "category_1", "keyword": "做什么都没兴趣"},
    {"text": "您是否经常感到心情低落？", "category": "category_1", "keyword": "心情低落"},
    {"text": "您是否经常感到疲劳或无精打采？", "category": "category_1", "keyword": "经常感到疲劳"},
    {"text": "您是否经常有不如死掉或用某种方式伤害自己的念头？", "category": "category_1", "keyword": "想自杀"},

    # 生理症状
    {"text": "您是否经常不太想吃东西，胃口不好？", "category": "category_2", "keyword": "不太想吃东西，胃口不好"},
    {"text": "您是否经常睡眠情况不好？", "category": "category_2", "keyword": "经常睡不好"},
    {"text": "您是否经常夜间睡眠不好？", "category": "category_2", "keyword": "夜间睡眠不好"},
    {"text": "您是否经常吃饭像平常一样多？", "category": "category_2", "keyword": "吃饭像平常一样多"},
    {"text": "您是否经常感到体重减轻？", "category": "category_2", "keyword": "体重减轻"},
    {"text": "您是否经常为便秘烦恼？", "category": "category_2", "keyword": "为便秘烦恼"},
    {"text": "您是否经常感到心跳比平时快？", "category": "category_2", "keyword": "心跳比平时快"},
    {"text": "您是否性功能正常？", "category": "category_2", "keyword": "性功能正常"},
    {"text": "您是否经常睡眠困难，很难熟睡或者睡太多？", "category": "category_2",
     "keyword": "睡眠困难，很难熟睡或者睡太多"},
    {"text": "您是否经常胃口不好或吃太多？", "category": "category_2", "keyword": "胃口不好或吃太多"},

    # 认知功能

    {"text": "您是否经常觉得自己的生活是失败的？", "category": "category_3", "keyword": "自我感觉成功"},
    {"text": "您是否经常觉得生活得很有意思？", "category": "category_3", "keyword": "生活有意思"},
    {"text": "您是否经常觉得前途是有希望的？", "category": "category_3", "keyword": "前途有希望"},
    {"text": "您是否经常觉得和其他人一样好？", "category": "category_3", "keyword": "和一般人一样好"},
    {"text": "您是否经常对未来感到有希望？", "category": "category_3", "keyword": "感觉未来有希望"},
    {"text": "您是否经常觉得决定什么事情很容易？", "category": "category_3", "keyword": "决定容易"},
    {"text": "您是否经常仍旧喜爱自己平时喜爱的东西？", "category": "category_3", "keyword": "仍旧喜爱"},
    {"text": "您是否经常感到自己是有用的和不可缺少的人？", "category": "category_3",
     "keyword": "我感觉我是有用且不可缺少"},
    {"text": "您是否经常觉得自己的生活很有意义？", "category": "category_3", "keyword": "生活有意义"},
    {"text": "您是否经常觉得自己很糟，或很失败，或让自己或家人失望？", "category": "category_3",
     "keyword": "感觉糟糕或失败"},

    # 社交关系

    {"text": "您是否有时说话比平时要少？", "category": "category_4", "keyword": "比平时说话少"},
    {"text": "您是否经常感到孤单？", "category": "category_4", "keyword": "感到孤单"},
    {"text": "您是否经常觉得别人不友善？", "category": "category_4", "keyword": "别人不友善"},
    {"text": "您是否经常觉得人们不喜欢您？", "category": "category_4", "keyword": "人们不喜欢您"},
    {"text": "您是否经常坐卧不安，难以保持平静？", "category": "category_4", "keyword": "坐卧不安"},
    {"text": "您是否经常觉得假若我死了别人会过得更好？", "category": "category_4",
     "keyword": "假若我死了别人会过得更好"},

    # 日常功能
    {"text": "您是否经常觉得无法继续您的日常工作？", "category": "category_5", "keyword": "无法继续日常工作"},
    {"text": "您是否经常在做事时无法集中自己的注意力？", "category": "category_5", "keyword": "无法集中注意力"},
    {"text": "您的头脑是否经常像往常一样清晰？", "category": "category_5", "keyword": "头脑不清晰"},
    {"text": "您是否经常做事情像平时一样不感到困难？", "category": "category_5", "keyword": "做事困难"},
    {"text": "您是否经常注意很难集中，例如阅读报纸或看电视？", "category": "category_5", "keyword": "注意难集中"},
    {"text": "您是否经常动作或说话速度缓慢到别人可察觉的程度，或者正好相反—您烦躁或坐立不安，动来动去的情况比平常更严重？",
     "category": "category_5", "keyword": "说话速度异常"},
    {"text": "以上这些问题在您工作、处理家庭事务，或与他人相处上经常造成多大的困难？", "category": "category_5",
     "keyword": "以上这些问题给我造成了很大困难"}

]

max_number = {
    "category_1": 5, "category_2": 5, "category_3": 5, "category_4": 2, "category_5": 2
}
now_number = {
    "category_1": 0, "category_2": 0, "category_3": 0, "category_4": 0, "category_5": 0
}


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
    sentence_embeddings = get_sentence_embeddings([sent["text"] for sent in sentences])

    # 用户输入的嵌入
    user_input_embedding = get_sentence_embeddings([user_input])

    # 计算用户输入与每个句子之间的余弦相似度
    similarities = cosine_similarity(user_input_embedding, sentence_embeddings)

    # 获取相似度最高的句子的索引
    top_k_indices = similarities.argsort(axis=1)[0][-top_k:][::-1]

    # 返回与用户输入最相关的句子，并将其添加到已搜索列表中
    related_sentences = [sentences[i] for i in top_k_indices]

    return related_sentences


# 找到文本所在类别
def find_category(sentence, sentences_by_category):
    for item in sentences_by_category:
        if item["text"] == sentence:
            return item["category"]


# 功能：找到对应输出句子的category，1.将句子删除，2.category计数
def deal_category(sentence, sentences_by_category):
    for item in sentences_by_category:
        if item["text"] == sentence:
            cate = item["category"]
            now_number[cate] += 1
            # print("cate:", cate, "num:", now_number[cate])
            sentences_by_category.remove(item)
            return


# 检查问卷中每个类别是否都问完了
def check_max_number_reached(now_number, max_number):
    for category, now_count in now_number.items():
        if now_count < max_number.get(category, 0):
            # 如果某个类别的当前数量小于对应的最大数量，则返回 False
            return False
    # 所有类别的当前数量都大于或等于对应的最大数量，则返回 True
    return True


def get_next_ques(user_ans):
    # 找下一个问题
    related_sentence = search_related_sentence(sentences_by_category, user_ans)
    now_cate = find_category(related_sentence[0]["text"], sentences_by_category)
    if related_sentence and now_number[now_cate] < max_number[now_cate]:
        # 如果找到了，并且该类别还没有达到最大个数，则输出
        now_ques = related_sentence[0]["text"]
        # print("下一个问题：", print_sentence)
        deal_category(now_ques, sentences_by_category)

    else:
        # 如果没有找到相关句子，就随机选择一个未输出的句子输出
        # 随机找的句子所处类别应还没达到最大输出数量
        not_searched_sentences = [sent["text"] for sent in sentences_by_category
                                  if now_number[find_category(sent["text"], sentences_by_category)] <
                                  max_number[find_category(sent["text"], sentences_by_category)]]

        random_sentence = random.choice(not_searched_sentences)
        now_ques = random_sentence
        deal_category(now_ques, sentences_by_category)
    return now_ques


# 初始化
now_ques = "您是否经常因为一些小事而烦恼？"
total_score = 0
# 多轮问答
while True:
    print("next question:", now_ques)
    user_input = input("请输入您的回答：")
    model_input = now_ques + user_input
    now_prom = get_prompt(now_ques)
    total_score += int(rank(model_input, now_prom))
    reply = chat_request(model_input)
    print(reply)

    # 用户回答的隐性评分
    if implicit(user_input, sentences_by_category):
        im_ques = implicit(user_input, sentences_by_category)
        ques_ans = im_ques + user_input
        now_prom = get_prompt(im_ques)
        total_score += int(rank(ques_ans, now_prom))
        deal_category(im_ques, sentences_by_category)
        
    # 搜索下一个问题
    now_ques = get_next_ques(user_input)

    # 所有类别的问题个数都达到了
    if check_max_number_reached(now_number, max_number):
        print("问卷结束～")
        print("final score is:", total_score)
        break
