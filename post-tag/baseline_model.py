# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:14:59 2018

@author: rizkyfalih
"""

def read_dataset(fname):
    sentences = []
    tags = []
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    idx_line = 0
    while idx_line < len(content):
        sent = []
        tag = []
        while not content[idx_line].startswith('</kalimat'):
            if  not content[idx_line].startswith('<kalimat'):
                content_part = content[idx_line].split('\t')
                sent.append(content_part[0])
                tag.append(content_part[1])
            idx_line = idx_line + 1
        sentences.append(sent)
        tags.append(tag)
        idx_line = idx_line+2        
    return sentences, tags

def accuracy(predicted_list, expected_list):
    import numpy as np
    score = np.zeros(len(predicted_list))
    for i, sentence in enumerate(expected_list):
        for j, tag in enumerate(predicted_list[i]):
            if (tag == expected_list[i][j]):
                score[i] += 1
        score[i] /= len(sentence)
        
    return sum(score)/20

sentences,tags = read_dataset('Indonesian_Manually_Tagged_Corpus_ID.tsv')
train_word = sentences[:1000]
train_tag = tags[:1000]
test_word = sentences[1000:1020]
test_tag = tags[1000:1020]
tagDict = {}
length = len(train_word)

# Baseline
for i in range(0, length - 1):
    for j in range(0, len(train_word[i])-1):
        if train_word[i][j].lower() in tagDict:
            if train_tag[i][j].lower() in tagDict[train_word[i][j].lower()]:
                tagDict[train_word[i][j].lower()][train_tag[i][j]] += 1
            else:
                tagDict[train_word[i][j].lower()][train_tag[i][j]] = 1 
        else:
            tagDict[train_word[i][j].lower()] = {}
            tagDict[train_word[i][j].lower()][train_tag[i][j]] = 1 
        
# Predict
sentence_pred = []
for i in range(0, len(test_word)):
    word_pred = []
    for j in range(0, len(test_word[i])):
        try:
            tag_pred = max(tagDict[test_word[i][j].lower()], key=tagDict[test_word[i][j].lower()].get)
            word_pred.append(tag_pred)
        except:
            tag_pred = 'NN'
            word_pred.append(tag_pred)
    sentence_pred.append(word_pred)
    
print(accuracy(sentence_pred, test_tag))