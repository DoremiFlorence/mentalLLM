# -*- coding: utf-8 -*-
# @Time : 5/17/24 11:24 AM
# @Author : Florence
# @File : match.py
# @Project : llm


def implicit(user_answer, cate_sent):
    # 在用户回答中搜索关键词
    matched_question = None
    for sent in cate_sent:
        if sent["keyword"] in user_answer:
            matched_question = sent["text"]
            break

    if matched_question:
        return matched_question
        # print("匹配到的问题：", matched_question)
    else:
        return False
        # print("未匹配到相关问题。")



