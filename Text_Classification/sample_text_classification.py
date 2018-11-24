#source: answer from https://stackoverflow.com/questions/39121104/how-to-add-another-feature-length-of-text-to-current-bag-of-words-classificati

import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer

X_train = np.array(["khas banget dan enak banget trus berasa mandi pake sabun mahal lah pokoknya wkwk..",
                    "Sabun mandi klasik dan berasa naik pangkat kalau pake ini. Kenapa? Karena ngerasa keren aja beb kek berasa sabun mahal. Wanginya klasik, gak wangi floral atau apa yang familiar gitu. So far suka pakenya karena ada experience tertentu yang didapat, selain mandi jadi bersih dan wangi ya. Sabun memori lah pokoknya ?",
                    "Literally beli ini karena mupeng sama baunya, baunya itu ya wagelasehh, enak parah. Haha awalnya aku kira ini sabun pricey soalnya judulnya aja udah pake 'imperial'. Efeknya kurang lbh sama kaya sabun mandi pada umumnya cuma baunya yg khas itu nilai plus buatku. Repurchase? yes klo sabun rumah udah abis.",
                    "Dulu kalo beli sabun ini pasti kesannya mewah banget. Selain wanginya yang khas dan lembut, bentuk sabunnya juga premium banget. Suka lama-lama mandi kalo pake sabun ini karena wanginya bikin nyaman banget. Bikin lembab juga. Enak deh pake sabun ini",
                    "Sabun ini keras jadi awet dan nggak cepet abis. Wanginya aku kurang suka. Busanya jg nggak terlalu banyak, aku suka yg banyak busanya biar kerasa bersih. Tapi nggak bikin kulit kering sih. Harganya pun terjangkau",
                    "welehh...ini sabun jaman dulu yang masih eksis Sukanya: ~ awet banget soalnya keras gitu sabunnya, ga abis-abis ~ ga terlalu berbusa Ga sukanya: ~ wanginya ga suka... ~ ga ngelembabin kulit ~ susah megangnya karena terlalu keras",
                    "Dulu banget sebelum mengenal sabun cair, aku pakai sabun batangan merk ini. Tapi aku kurang suka dengan aromanya. Busanya kurang banyak. Tidak cocok dikulit aku yang sensitif. Karena tiap kali aku mandi menggunakan sabun ini, setelah handukan kulit ku berasa gatal-gatal.",
                    "Imperial Leather nih dulu terkenal awet sabunnya. Wanginya lembut. Tapi pas ku coba, ga ada awet-awetnya. Sama aja kayak bar soap biasanya. Wanginya memang lembut. Sesaat setelah bilas, wanginya sudah langsung hilang seketika. Jadi bagi aku, nothing special dengan sabun ini.",])
y_train = np.array([[1],[1],[1],[1],[0],[0],[0],[0]])

X_test = np.array(["ga suka, sabun ini keras",
                   "suka baunya, mewah."
                   ])   
target_names = ['Class 1', 'Class 2']


def get_text_length(x):
    return np.array([len(t) for t in x]).reshape(-1, 1)

# hanya menggunakan fitur berupa vektor tf-idf
classifier1 = Pipeline([
    ('vectorizer', CountVectorizer(min_df=1,max_df=2)),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

# menggunakan fitur gabungan vektor tf-idf dan panjang teks
classifier2 = Pipeline([
    ('features', FeatureUnion([
        ('text', Pipeline([
            ('vectorizer', CountVectorizer(min_df=1,max_df=2)),
            ('tfidf', TfidfTransformer()),
        ])),
        ('length', Pipeline([
            ('count', FunctionTransformer(get_text_length, validate=False)),
        ]))
    ])),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier2.fit(X_train, y_train)
predicted = classifier2.predict(X_test)
print(predicted)