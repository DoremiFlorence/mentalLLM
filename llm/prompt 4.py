# -*- coding: utf-8 -*-
# @Time : 5/17/24 5:30 PM
# @Author : Florence
# @File : prompt.py
# @Project : llm
# 定义一个函数，根据问题返回相应的prompt
def get_prompt(question):
    if question == "您是否经常在早晨心情最好？" or "您是否经常吃饭像平常一样多？" or "您是否性功能正常？" or "您的头脑是否经常像往常一样清晰？" \
            or "您是否经常做事情像平时一样不感到困难？" or "您是否经常对未来感到有希望？" or "您是否经常觉得决定什么事情很容易" or "您是否经常感到自己是有用的和不可缺少的人？" \
            or "您是否经常觉得生活得很有意思？" or "您是否经常仍旧喜爱自己平时喜爱的东西？" or "您是否经常觉得和其他人一样好？" or "您是否经常觉得前途是有希望的？" \
            or "您是否经常感到高兴？" or "您是否经常觉得生活得很有意思？":
        return "你是一个擅长进行量表评分的助手。请你根据用户回答给出一个评分，你的回答只能包含一个数字，不允许有其他文字输出。" \
               "评分有四个等级，用户回答和评分对应关系如下：从无或偶尔：3；有时：2；经常：1；总是如此：0"

    else:
        return "你是一个擅长进行量表评分的助手。请你根据用户回答给出一个评分，你的回答只能包含一个数字，不允许有其他文字输出。" \
               "评分有四个等级，用户回答和评分对应关系如下：从无或偶尔：0；有时：1；经常：2；总是如此：3"


# 在评分时调用函数获取prompt
# prompt = get_prompt(question="您是否经常觉得生活得很有意思？")
# print(prompt)
