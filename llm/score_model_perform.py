# -*- coding: utf-8 -*-
# @Time : 6/9/24 11:04 AM
# @Author : Florence
# @File : score_model_perform.py
# @Project : llm
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from score2 import rank, get_prompt
import matplotlib.pyplot as plt
import re


# 假设你的模型是一个函数，名为`predict_score`
# 它接受两个参数：question 和 answer，并返回一个分数
def predict_score(question, answer):
    # 这里应该是你模型的实际实现
    # 例如：return model.predict([question, answer])
    model_input = answer
    prom = get_prompt(question)
    result = rank(model_input, prom)
    return result

def extract_number_from_output(output):
    numbers = re.findall(r'\d+', output)
    # 假设你只关心第一个匹配的数字，并将其转换为整数
    if numbers:
        number = int(numbers[0])
        # 检查提取的数字是否在0到3之间
        if 0 <= number <= 3:
            return number
        # 如果没有数字或数字不在0到3之间，返回1
    return 1
    # return int(numbers[0]) if numbers else None

# 读取数据集
data = pd.read_csv('test_dataset.csv')

# 获取问题、答案和实际分数
questions = data.iloc[:, 0].values
answers = data.iloc[:, 1].values
actual_categories = data.iloc[:, 2].astype(int).values
# 使用模型预测分数
# predicted_scores = [predict_score(q, a) for q, a in zip(questions, answers)]
predicted_scores = [extract_number_from_output(predict_score(q, a)) for q, a in zip(questions, answers)]

# 将预测分数转换为浮点数
predicted_scores = [float(score) for score in predicted_scores]

# 计算评估指标
mse = mean_squared_error(actual_categories, predicted_scores)
mae = mean_absolute_error(actual_categories, predicted_scores)
r2 = r2_score(actual_categories, predicted_scores)

# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"R-squared (R2): {r2}")


# 计算分类评估指标
accuracy = accuracy_score(actual_categories, predicted_scores)
report = classification_report(actual_categories, predicted_scores,
                               target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
conf_matrix = confusion_matrix(actual_categories, predicted_scores)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)
