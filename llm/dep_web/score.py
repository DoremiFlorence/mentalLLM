# -*- coding: utf-8 -*-
# @Time : 5/17/24 6:02 PM
# @Author : Florence
# @File : score.py
# @Project : llm

import requests
import json


def rank(question, prompt):
    url = "http://10.50.0.35:7000/v1/chat/completions"
    # 创建一个符合ChatCompletionRequest类结构的字典
    data = {
        "model": "chatglm3-6b",
        "messages": [
            {
                "role": "system",
                "content": prompt
            },

            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }

    # 将字典转换为JSON格式
    json_data = json.dumps(data)

    times = 0
    while True:
        # 发送POST请求
        response = requests.post(url, data=json_data)

        # 获取响应JSON
        response_json = response.json()

        # 获取并打印content
        content = response_json['choices'][0]['message']['content']

        # 检查content是否为有效数字
        if content.isdigit() and 0 <= int(content) <= 3:
            # 如果是有效数字，则返回content
            return content
        else:
            # 如果不是有效数字，则再次调用模型
            times += 1
            if times > 10:
                return 1
            continue
