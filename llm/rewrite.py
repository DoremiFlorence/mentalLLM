# -*- coding: utf-8 -*-
# @Time : 4/13/24 4:06 PM
# @Author : Florence
# @File : rewrite.py
# @Project : llm
import requests
import json

def in_rewrite(question):
    url = "http://10.50.0.35:8000/v1/chat/completions"
    # 创建一个符合ChatCompletionRequest类结构的字典
    data = {
        "model": "chatglm3-6b",
        "messages": [
            {
                "role": "system",
                "content":  "你是一个擅长进行总结的助手，我将给你一个心理测试问句和一个用户回答，请你根据这对问答，提炼出用户的情况总结。"
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": 0.5,
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