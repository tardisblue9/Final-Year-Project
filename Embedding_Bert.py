#encoding=utf-8
import numpy as np
import gensim
from gensim.models import KeyedVectors
import torch;
import re

import tensorflow as tf;
import torch;
import transformers;
from transformers import AutoModel, AutoTokenizer


def split(word):
    return [char for char in word] 

def divide_text_by_sentence(text):
    text = text.strip('\n')
    text = re.split('？|。|！|……|：|“|”',text) # ，|
    return text # list

def get_sentence_bert(sent,tokenizer,model):
    input_ids = tokenizer.encode(sent, return_tensors='pt')
    last_hidden_state, _ = model(input_ids)
    return last_hidden_state

corpus_os = '/Users/vanellope/Desktop/FYP/textbook_corpus/textbook_HK_6class.txt'
file = open(corpus_os,encoding='utf-8')
texts = file.readlines()
file.close()
temp = []
for text in texts:
    temp.append(split(text))
texts = temp


tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')

##############################
print('starting to process the corpus')
file = open(corpus_os)
tb = file.readlines()
file.close()

tb = tb[1:]
print(tb[10],'\n',len(tb))

# loop all article
textbook_vec = []
for i,text in enumerate(tb):
    article = []
    text = divide_text_by_sentence(text)
    for sent in text:
        if len(sent) == 0:
            continue # drop empty sentence
        vec = get_sentence_bert(sent,tokenizer,model)
        article.append(vec)
    # print(len(article)) # 每篇文章句子数
    textbook_vec.append(article)
    
    torch.save(textbook_vec,'/Users/vanellope/2020-2021 Final Year Project/FYP_BERT_CORPUS_6CLASS_HK/article{}'.format(i))
    textbook_vec = []


print('lets load one article:')
article_batch = torch.load('/Users/vanellope/2020-2021 Final Year Project/FYP_BERT_CORPUS_6CLASS_HK/article10')
article = article_batch[0]
print(len(article))
for i in range(len(article)):
    print(article[i].shape)
