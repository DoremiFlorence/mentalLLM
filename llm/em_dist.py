# -*- coding: utf-8 -*-
# @Time : 6/10/24 5:56 PM
# @Author : Florence
# @File : em_dist.py
# @Project : llm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import os
import random
import numpy as np
import pandas as pd

# 设置模型路径
MODEL_PATH = os.environ.get('MODEL_PATH', '/home/zhengjinfang/bge-large-zh-v1.5')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

# 加载模型和标记器
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval()

# 定义正向和反向的频率描述及其对应的评分
positive_frequency_descriptions = {
    "总是": 3,
    "几乎每天": 3,
    "经常": 2,
    "一半的时间": 2,
    "有时": 1,
    "偶尔": 1,
    "从未": 0,
    "很少": 0,
}

negative_frequency_descriptions = {
    "总是": 0,
    "几乎每天": 0,
    "经常": 1,
    "一半的时间": 1,
    "偶尔": 2,
    "有时": 2,
    "从未": 3,
    "很少": 3,
}


# 将频率描述转换为向量
def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


# 编码正向和反向的频率描述
positive_frequency_vectors = {desc: encode_text(desc) for desc in positive_frequency_descriptions.keys()}
negative_frequency_vectors = {desc: encode_text(desc) for desc in negative_frequency_descriptions.keys()}


# 计算输入文本的评分
def get_score(input_text, direction='positive'):
    input_vector = encode_text(input_text)
    if direction == 'positive':
        frequency_vectors = positive_frequency_vectors
        frequency_descriptions = positive_frequency_descriptions
    else:
        frequency_vectors = negative_frequency_vectors
        frequency_descriptions = negative_frequency_descriptions

    similarities = {desc: cosine_similarity(input_vector, vec)[0][0] for desc, vec in frequency_vectors.items()}
    best_match = max(similarities, key=similarities.get)
    return frequency_descriptions[best_match]


def get_prompt(question):
    if question in ["您是否经常在早晨心情最好？", "您是否经常吃饭像平常一样多？", "您是否性功能正常？",
                    "您的头脑是否经常像往常一样清晰？",
                    "您是否经常做事情像平时一样不感到困难？", "您是否经常对未来感到有希望？",
                    "您是否经常觉得决定什么事情很容易",
                    "您是否经常感到自己是有用的和不可缺少的人？", "您是否经常觉得生活得很有意思？",
                    "您是否经常仍旧喜爱自己平时喜爱的东西？",
                    "您是否经常觉得和其他人一样好？", "您是否经常觉得前途是有希望的？", "您是否经常感到高兴？",
                    "您是否经常觉得生活得很有意思？"]:
        return "negative"

    else:
        return "positive"


# 读取数据集
data = pd.read_csv('test_dataset.csv')

# 获取问题、答案和实际分数
questions = data.iloc[:, 0].values
answers = data.iloc[:, 1].values
actual_categories = data.iloc[:, 2].astype(int).values

# p_n = get_prompt(questions)
# 使用语义检索计算预测分数
predicted_categories = []
for question, answer in zip(questions, answers):
    direction = get_prompt(question)
    score = get_score(answer, direction)
    predicted_categories.append(score)

# 转换为整数
predicted_categories = [int(score) for score in predicted_categories]

# 计算分类评估指标
accuracy = accuracy_score(actual_categories, predicted_categories)
report = classification_report(actual_categories, predicted_categories,
                               target_names=[f'Class {i}' for i in sorted(set(actual_categories))])
conf_matrix = confusion_matrix(actual_categories, predicted_categories, labels=sorted(set(actual_categories)))
mse = mean_squared_error(actual_categories, predicted_categories)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# 保存结果
data['Predicted'] = predicted_categories
# data.to_csv('predicted_results.csv', index=False)
