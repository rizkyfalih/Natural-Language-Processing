# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk

freq_tab = {}
par = "Pesilat Indonesia kembali meraih medali emas ke-12, setelah Pipiet Kamelia berhasil menumbangkan pesilat Vietnam Thi Cam Nhi Nguyen di babak final pertandingan cabang olahraga Pencak Silat di ajang Asian Games 2018."
par += 'Pipiet menang telak 5-0 atas Thi Cam di kelas D putri 60kg-65 kg, yang berlangsung di Padepokan Pencak Silat Taman Mini Indonesia Indah (TMII), Jakarta Timur, Rabu petang. '
par += 'Dengan kemenangan ini, Pipiet berhasil membawa medali emas, sementara Thi Cam harus puas juara kedua dengan medali perak. '

lc_par = par.lower()
par_tokens = nltk.word_tokenize(lc_par)

freq_tab = {}
total_count = 0
for token in par_tokens:
    if token in freq_tab:
        freq_tab[token] +=1
    else:
        freq_tab[token] = 1
    total_count +=1

prob_tab = {}
for token in freq_tab:
    prob_tab[token] = freq_tab[token]/total_count
    
print(prob_tab)


test_sentence  = 'Pesilat Indonesia meraih emas'
lc_test_sentence = test_sentence.lower()
test_tokens = nltk.word_tokenize(lc_test_sentence)
total_prob = 1.0
for test_token in test_tokens:
    total_prob = total_prob * prob_tab[test_token]
    print(test_token)
    
print(total_prob)