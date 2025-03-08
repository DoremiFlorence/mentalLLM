# from gensim.models import KeyedVectors
# import numpy as np
#
# # 加载预训练的GloVe模型或Word2Vec模型
# model = KeyedVectors.load_word2vec_format('path_to_model.bin', binary=True)
# sentence1 = "This is a test sentence."
# sentence2 = "This is another sentence."
#
# def get_sentence_embedding(sentence, model):
#     words = sentence.lower().split()
#     word_vectors = [model[word] for word in words if word in model]
#     return np.mean(word_vectors, axis=0)
#
# embedding1 = get_sentence_embedding(sentence1, model)
# embedding2 = get_sentence_embedding(sentence2, model)
# cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity
import torch

from transformers import BertTokenizer, BertModel

# 指定本地路径
local_tokenizer_path = '/Users/florence/Downloads/bert-base-uncased'
local_model_path = '/Users/florence/Downloads/bert-base-uncased'

# 从本地加载 tokenizer
tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path)

# 从本地加载模型
model = BertModel.from_pretrained(local_model_path)

# 输入句子对
sentence1 = "This is a test sentence."
sentence2 = "This is another test sentence."

# 对输入句子进行tokenization
inputs = tokenizer(sentence1, sentence2, return_tensors='pt', padding=True, truncation=True)

# 通过BERT模型获取嵌入
outputs = model(**inputs)

# 提取 [CLS] 标记的嵌入
cls_embedding = outputs.last_hidden_state[:, 0, :]

# 计算余弦相似度
similarity = cosine_similarity(cls_embedding[0], cls_embedding[1], dim=0)

print("Cosine similarity:", similarity.item())
