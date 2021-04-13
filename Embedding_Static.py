#encoding=utf-8
import numpy as np
import gensim
from gensim.models import KeyedVectors
import torch;
import re


def split(word):
    return [char for char in word] 

def divide_text_by_sentence(text):
    text = text.strip('\n')
    text = re.split('？|。|！|……|：|“|”',text) # ，|
    return text # list

def get_sentence_stat(sent,wv):
    words = split(sent)
    t = []
    for word in words:
        try:
            t.append(wv[word])
        except:# some words are not in vocabulary
            continue
    t = np.array(t)
    return torch.tensor([t])

def get_texts(corpus_os):
    file = open(corpus_os,encoding='utf-8')
    texts = file.readlines()
    file.close()
    temp = []
    for text in texts:
        temp.append(split(text))
    texts = temp
    return texts

def train_save_static_model(embedding_dimension,window_size,texts):
    # word2vec model
    model = gensim.models.Word2Vec(size=embedding_dimension, window=window_size, min_count=1)  # instantiate
    model.build_vocab(sentences=texts)
    model.train(sentences=texts, total_examples=len(texts), epochs=10)  # train
    word_vectors = model.wv
    word_vectors.save("/Users/vanellope/Desktop/FYP/textbook_corpus/HK_word2vec_{}.wordvectors".format(embedding_dimension))
    print('embedding model is saved locally')
    return

def save_embed_texts(OS_embedding_dimension,corpus_os):
    print('loading local model')
    wv = KeyedVectors.load("/Users/vanellope/Desktop/FYP/textbook_corpus/HK_word2vec_{}.wordvectors".format(OS_embedding_dimension), mmap='r')
    # print("have a look at similar words to {} :".format('雨'),wv.most_similar(['雨']))

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
            vec = get_sentence_stat(sent,wv) 
            article.append(vec)
        # print(len(article)) # 每篇文章句子数
        textbook_vec.append(article)
        
        torch.save(textbook_vec,'/Users/vanellope/2020-2021 Final Year Project/FYP_Static_CORPUS_6CLASS_HK/article{}'.format(i))
        textbook_vec = []


    print('lets load one article:')
    article_batch = torch.load('/Users/vanellope/2020-2021 Final Year Project/FYP_Static_CORPUS_6CLASS_HK/article10')
    article = article_batch[0]
    print(len(article))
    for i in range(len(article)):
        print(article[i].shape)
        
    return

############## MAIN ################

corpus_os = '/Users/vanellope/Desktop/FYP/textbook_corpus/textbook_HK_6class.txt'
texts = get_texts(corpus_os)
train_save_static_model(embedding_dimension = 40,window_size = 5,texts=texts)
save_embed_texts(OS_embedding_dimension=40,corpus_os=corpus_os)





