import os
import faiss
import numpy as np
from req import chat_request
import torch
from transformers import AutoTokenizer, AutoModel

# 加载模型和标记器
MODEL_PATH = os.environ.get('MODEL_PATH', '/home/zhengjinfang/bge-large-zh-v1.5')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).eval()


def get_sentence_embeddings(sentences):
    # 使用模型和标记器编码句子
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取每个句子的隐藏状态
    sentence_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return sentence_embeddings


# 1. 准备知识库文档
documents = []
docs_file = '/home/zhengjinfang/knowledge.txt'  # 文件路径
with open(docs_file, 'r', encoding='utf-8') as file:
    documents = file.readlines()

# 2. 文本预处理和向量化
doc_embeddings = get_sentence_embeddings(documents)

# 3. 构建FAISS索引
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)


# 4. 检索和生成答案
def retrieve_and_generate(question, top_k=3):
    # 问题转化为向量
    question_embedding = get_sentence_embeddings([question])
    print("11111")

    # 使用FAISS检索相关文档
    distances, indices = index.search(question_embedding, top_k)
    print("distance: ",distances)
    retrieved_docs = [documents[i] for i in indices[0]]

    print("22222")
    # 将检索到的文档和问题一起输入ChatGLM3生成答案
    context = "\n".join(retrieved_docs)
    print("######", context)
    input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    print("******", input_text)
    # input_text=question
    reply = chat_request(input_text)

    return reply


# 示例问题
question = "为什么心情不好？"
print("que:", question)
answer = retrieve_and_generate(question)
print(answer)
