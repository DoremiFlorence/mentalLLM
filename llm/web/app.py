from flask import Flask, render_template, request, jsonify
from req1 import chat_request

app = Flask(__name__)

# 创建问题序列
questions = [
    "您是否觉得做任何事情都没有兴趣或乐趣？",
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


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_question', methods=['POST'])
def get_question():
    data = request.get_json()
    question_index = data.get('questionIndex', 0)
    if question_index < len(questions):
        return jsonify({'question': questions[question_index]})
    else:
        return jsonify({'question': None})  # No more questions


@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    answer = data.get('answer', '')
    question_index = data.get('questionIndex', 0)
    reply = process_answer(answer, question_index)
    return jsonify({'reply': reply})


def process_answer(answer, question_index):
    # 逻辑处理用户的回答
    if question_index < len(questions):
        next_question = questions[question_index]
        reply = chat_request(f"{next_question}{answer}")
        return reply


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
