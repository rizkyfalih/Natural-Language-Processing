# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 09:47:26 2018

@author: rizkyfalih
"""

def read_train(fname):
    tag_count = {}
    word_tag = {}
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    idx_line = 0
    
    while idx_line < len(content):
        while not content[idx_line].startswith('</kalimat'):
            if  not content[idx_line].startswith('<kalimat'):
                content_part = content[idx_line].split('\t')
                if content_part[1] in tag_count:
                    tag_count[content_part[1]] += 1
                else:
                    tag_count[content_part[1]] = 1
                    
                current_word_tag = content_part[0]+','+content_part[1]
                if current_word_tag in word_tag:
                    word_tag[current_word_tag] += 1
                else:    
                    word_tag[current_word_tag] = 1

            idx_line = idx_line + 1

        idx_line = idx_line+1
        
    return tag_count, word_tag

def read_test(fname):
    with open(fname) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    sentence = []
    sentences = []
    idx_line = 0

    while idx_line < len(content):
        while not content[idx_line].startswith('</kalimat'):
            if  content[idx_line].startswith('<kalimat'):
                sentence = []
            if  not content[idx_line].startswith('<kalimat'):
                content_part = content[idx_line].split('\t')
                sentence.append(content_part)
            idx_line = idx_line + 1
        sentences.append(sentence)
            
        idx_line = idx_line+1
    
    return sentences

def baseline(word_tag, sentence):
    s = [x[0] for x in sentence]
    expected_tag = [x[1] for x in sentence]
    tag_sequence = []
    
    for i, word in enumerate(s):
        scores = []
    
        for j, key in enumerate(word_tag.keys()):
            k = key.split(',')[0].lower()
            tag = key.split(',')[1]
            
            if (word == k):
                scores.append({ 'k': k, 'tag': tag, 'score': word_tag[key] })
        
        onlyScores = [x['score'] for x in scores]
        
        try:
            max_index = onlyScores.index(max(onlyScores))
            best_tag = scores[max_index]['tag']
            tag_sequence.append(best_tag)
        except:
            tag_sequence.append('NN')
        
    return tag_sequence, expected_tag

def baseline_predict(test_list):
    expected_list = []
    predicted_list = []
    for i, sentence in enumerate(test_list):
        tagPredicted, tagExpected = baseline(word_tag, sentence)
        expected_list.append(tagExpected)
        predicted_list.append(tagPredicted)

    return expected_list, predicted_list

def accuracy(predicted_list, expected_list):
    import numpy as np
    Score = np.zeros(len(predicted_list))
    for i, sentence in enumerate(expected_list):
        for j, tag in enumerate(predicted_list[i]):
            if (tag == expected_list[i][j]):
                Score[i] += 1
        Score[i] /= len(sentence)
        
    return sum(Score)/20


# =============================================================================
# Main Program     
# =============================================================================

# get word_tag from data train
tag_count, word_tag = read_train('data_train.txt')

# get list of sentence to test
test_list = read_test('data_test.txt')

# predict the test_list
expected_list, predicted_list = baseline_predict(test_list)

# get test accuracy    
print(accuracy(predicted_list, expected_list))