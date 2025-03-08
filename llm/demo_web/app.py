# app.py
from flask import Flask, request, jsonify,render_template,send_from_directory
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import os
from req import chat_request
from match import implicit
from score import rank
from prompt import get_prompt
import random

app = Flask(__name__, static_folder='static')
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
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    sentence_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return sentence_embeddings


def search_related_sentence(sentences, user_input, top_k=1):
    sentence_embeddings = get_sentence_embeddings([sent["text"] for sent in sentences])
    user_input_embedding = get_sentence_embeddings([user_input])
    similarities = cosine_similarity(user_input_embedding, sentence_embeddings)
    top_k_indices = similarities.argsort(axis=1)[0][-top_k:][::-1]
    related_sentences = [sentences[i] for i in top_k_indices]
    return related_sentences


def find_category(sentence, sentences_by_category):
    for item in sentences_by_category:
        if item["text"] == sentence:
            return item["category"]


def deal_category(sentence, sentences_by_category):
    for item in sentences_by_category:
        if item["text"] == sentence:
            cate = item["category"]
            now_number[cate] += 1
            sentences_by_category.remove(item)
            return


def check_max_number_reached(now_number, max_number):
    for category, now_count in now_number.items():
        if now_count < max_number.get(category, 0):
            return False
    return True

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data['user_input']
    now_ques = data['now_ques']
    total_score = data['total_score']

    # 这里插入处理对话逻辑的代码
    model_input = now_ques + user_input
    now_prom = get_prompt(now_ques)
    current_score = int(rank(model_input, now_prom))
    total_score += current_score
    reply = chat_request(model_input)

    if implicit(user_input, sentences_by_category):
        im_ques = implicit(user_input, sentences_by_category)
        ques_ans = im_ques + user_input
        now_prom = get_prompt(im_ques)
        total_score += int(rank(ques_ans, now_prom))
        deal_category(im_ques, sentences_by_category)

    related_sentence = search_related_sentence(sentences_by_category, user_input)
    now_cate = find_category(related_sentence[0]["text"], sentences_by_category)
    if related_sentence and now_number[now_cate] < max_number[now_cate]:
        now_ques = related_sentence[0]["text"]
        deal_category(now_ques, sentences_by_category)
    else:
        not_searched_sentences = [sent["text"] for sent in sentences_by_category
                                  if now_number[find_category(sent["text"], sentences_by_category)] <
                                  max_number[find_category(sent["text"], sentences_by_category)]]
        random_sentence = random.choice(not_searched_sentences)
        now_ques = random_sentence
        deal_category(now_ques, sentences_by_category)

    if check_max_number_reached(now_number, max_number):
        return jsonify({"reply": reply, "next_question": None, "total_score": total_score, "finished": True})

    return jsonify({"reply": reply, "next_question": now_ques, "total_score": total_score, "finished": False})


# @app.route('/<path:filename>')
# def serve_static(filename):
#     return send_from_directory('.', filename)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
