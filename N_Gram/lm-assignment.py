# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import the libraries
import pandas as pd
import re
import nltk

# preprocess to remove numerical/special characters
def preprocess(data):
    newCorpus=""
    
    for i in data.index:
        berita = re.sub('[!"\',]', r' \ ', str(dataset.loc[i, 'Berita']))
        title = re.sub('[!"\',]', r' \ ', str(dataset.loc[i, 'Title']))
        berita = re.sub('[^a-zA-Z!"\',]', ' ', str(berita))
        title = re.sub('[^a-zA-Z!"\',]', ' ', str(title))
        berita = berita.lower()
        title = title.lower()
        berita = berita.split()
        title = title.split()
        berita = ' '.join(berita)
        title = ' '.join(title)
        corpus = title + " " + berita + " "
        newCorpus += corpus
    return newCorpus

def unigram(tokens):
    unigramList = {}
    unigramProb = {}
    unigramCount = 0
    
    # create unigram lists
    for token in tokens:
        if token in unigramList:
            unigramList[token] += 1
        else:
            unigramList[token] = 1
        unigramCount += 1
        
    # create prob_list for unigram
    for token in unigramList:
        unigramProb[token] = unigramList[token]/unigramCount
    
    return unigramList,unigramCount,unigramProb
    
def bigram(tokens):
    bigramList = {}
    bigramCount = 0
    length = len(tokens)
    for i in range(0, length - 1):
        if tokens[i] in bigramList:
            if tokens[i + 1] in bigramList[tokens[i]]:
                bigramList[tokens[i]][tokens[i + 1]] += 1
            else:
                bigramList[tokens[i]][tokens[i + 1]] = 1
        else:
            bigramList[tokens[i]] = {}
            bigramList[tokens[i]][tokens[i + 1]] = 1
            bigramCount += 1
    return bigramList,bigramCount

def probability_bigram(bigramList,uniList):
    bigramProb = {}
    for token in bigramList:
        bigramProb[token] = bigramList[token]
        for token2 in bigramList[token]:
            bigramProb[token][token2] = bigramList[token][token2]/uniList[token]
    return bigramProb

def prediction(array,biList):
    for i in range(len(array)):
        prediction = max(biList[array[i]], key=biList[array[i]].get)
        print("The Word is: " + array[i] + " - " + "The Next Word is: " + prediction)
        
def root(n, r):
    from numpy import roots
    return roots([1]+[0]*(r-1)+[-n])

def perplexity(word, biList):
    tokens = nltk.word_tokenize(word)
    prob = 1
    for i in range(len(tokens) - 1):
        prob = prob * biList[tokens[i]][tokens[i+1]] 
    prob = 1/prob
    if (len(tokens)-1) > 2:
        perplex = root(prob,len(tokens)-1)
        return perplex[len(tokens)-2]
    else:
        perplex = root(prob,len(tokens)-1)
        return perplex[0]

# =============================================================================
# Main Program
# =============================================================================

# Get data from a file
xl= pd.ExcelFile("dataSet.xlsx")
dataset = xl.parse("dataSet")

# initialize corpus
corpus = preprocess(dataset)

# create a token list from corpus
corpusToken = nltk.word_tokenize(corpus)

# create a unigram
uniList, uniCount, uniProb = unigram(corpusToken)

# create a bigram
biList, biCount = bigram(corpusToken) 

# change biList value to become probability value
biList= probability_bigram(biList,uniList)

# initialize 10 testing words
array_of_words = ['abdulgani', 'ac', 'akhir', 'prabowo', 'seorang', 'kota', 'jokowi', 'agama', 'kangkung', 'acara']

# predict the next word
prediction(array_of_words, biList)

# initialize 5 testing sentences
testList = ["beli kangkung pakai uang mainan animah gegerkan pasar", "masyarakat malang akan sakit jika wali kota jadi terpidana", "kpk tahan wali kota nonaktif malang", "segudang khasiat daun salam bagi kesehatan", "membangun start up tidak terbatas usia"]

# test the perplexity
for i in range(0, len(testList)):
    print("==========================================")
    perplex = perplexity(testList[i], biList)
    print(testList[i])
    print("perplexitynya adalah " + str(perplex))

