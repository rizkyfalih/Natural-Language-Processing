import nltk
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Exercise-1
# sentence = "The Operation began this morning and all the crews stay in the location"
sentence = "Pesilat Indonesia kembali meraih medali emas ke-12, setelah Pipiet Kamelia berhasil menumbangkan pesilat Vietnam Thi Cam Nhi Nguyen di babak final pertandingan cabang olahraga Pencak Silat di ajang Asian Games 2018. Pipiet menang telak 5-0 atas Thi Cam di kelas D putri 60kg-65 kg, yang berlangsung di Padepokan Pencak Silat Taman Mini Indonesia Indah (TMII), Jakarta Timur, Rabu petang. Dengan kemenangan ini, Pipiet berhasil membawa medali emas, sementara Thi Cam harus puas juara kedua dengan medali perak."
tokens = nltk.word_tokenize(sentence)
print(tokens)
print(len(tokens))

# Exercise-2
sentence  = sentence.lower()

# Exercise-3
tokens =  nltk.word_tokenize(sentence)
print(tokens)

# Exercise-4
stemmer = PorterStemmer()
for word in tokens:
    tempStem = stemmer.stem(word)
    print(tempStem)

# Exercise-5
wordLemma = WordNetLemmatizer()
for word in tokens:
    tempLemma = wordLemma.lemmatize(word)
    print(tempLemma)

# Excercise-6
tf_dict={}
total_count = 0
tf_dict['sentence'] = {}
for word in tokens:
    if word in tf_dict['sentence'].keys():
        tf_dict['sentence'][word] +=1
    else:
        tf_dict['sentence'][word] = 1
    total_count +=1

print(tf_dict)

prob_tab= {}
for token in tf_dict['sentence']:
    prob_tab[token] = tf_dict['sentence'][token]/total_count
print(prob_tab)

# # idf
# for key in tf_dict['sentence'].keys():
#     if key in idf_dict.keys():
#         idf_dict[key] += 1
#     else:
#         idf_dict[key] = 1


# for key[]