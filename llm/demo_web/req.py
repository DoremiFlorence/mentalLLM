# -*- coding: utf-8 -*-
# @Time : 5/17/24 6:03 PM
# @Author : Florence
# @File : req.py
# @Project : llm
import requests
import json


def chat_request(question):
    url = "http://10.50.0.35:8000/v1/chat/completions"
    # 创建一个符合ChatCompletionRequest类结构的字典
    data = {
        "model": "chatglm3-6b",
        "messages": [
            {
                "role": "system",
                "content": "接下来我会输入一个问题和一个对应的用户的答案，请你根据用户的答案给出回答。"
                           "对用户好的情绪和习惯给出鼓励。不好的习惯和情况要加以开导，给出一些可能的解决方法。尽量多说一些，用完整流畅的中文回答。"
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }

    # 将字典转换为JSON格式
    json_data = json.dumps(data)

    # 发送POST请求
    response = requests.post(url, data=json_data)

    # 获取响应JSON
    response_json = response.json()

    # 获取并打印content
    content = response_json['choices'][0]['message']['content']
    return content
