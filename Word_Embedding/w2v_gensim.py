import gensim
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        #untuk setiap file
        for fname in os.listdir(self.dirname):
            #untuk setiap baris
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

sentences = MySentences('./artikel') # a memory-friendly iterator

# sg = 0 -> CBOW, sg = 1 -> skip-gram
# size: dimensionality dari vektor kata yang dihasilkan
# min_count: banyaknya frekuensi miminal sebuah kata, jika ingin dipertimbangkan dalam proses
# window: range antara kata-kata konteks dengan posisi current word
model = gensim.models.Word2Vec(sentences, size=32, sg = 0, min_count = 1, window = 5, iter = 10)

# save model
model.save('./mymodel')

# load model
new_model = gensim.models.Word2Vec.load('./mymodel')

# mendapatkan representasi vektor dari sebuah kata
print (model.wv['dia'])

# menghitung similarity vektor antara dua kata
print (model.wv.similarity('kami', 'dia'))

# mencari top-N similar words
print (model.wv.similar_by_word('kami', topn=10, restrict_vocab=None))