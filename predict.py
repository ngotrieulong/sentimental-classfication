from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    my_model = load_model('model_file_doc2vec.h5')
    #my_model = load_model('model_cnn.h5')
    #my_model = load_model('model_file_bi-lstm.h5')
import csv
import numpy as np
import keras
def loaddic():
    dic1 = dict()
    with open('dic.csv', 'r') as csvFile:
        reader = csv.reader(csvFile)
        for word, _ in reader:
            dic1[word] = _
    csvFile.close()
    return dic1
max_length=303
dic2 = loaddic()
def newinput(sentences):
    data = list()  # list transform words to integer
    for i in sentences:
        t = i.split()
        data1 = []
        for word in t:  # count rank for every word in words
            index = dic2.get(word, 0)
            data1.append(index)
        data.append(data1)
    return data
negative=[]
positive=[]
def predict():
    with open('daura.csv', 'r', encoding='utf8') as file1:
        twt = []
        for row in file1:
            twt.append(row)
        twt = [i.replace('\n', "") for i in twt]
        twt = [i for i in twt if i != ""]
    print(twt)
    twt_updated = newinput(twt)
    twt_updated = np.asarray(twt_updated)
    # padding the tweet to have exactly the same shape as `embedding_2` input
    twt_updated = keras.preprocessing.sequence.pad_sequences(twt_updated, padding='post', maxlen=max_length)
    sentiment = my_model.predict(twt_updated)
    sentiment = np.argmax(sentiment, axis=1)
    # print('sentiment',sentiment)
    # print('y_test',y_test)
    for i in range(len(sentiment)):
        if sentiment[i] == 0 : negative.append(twt[i])
        else: positive.append(twt[i])
predict()
print('negative: ',negative)
print('positive: ',positive)
print('negative----------------------------------------------------------')
for i in negative:
    print('negative: ',i)
print('positive--------------------------------------------------------')
for i in positive:
    print('positive: ',i)
