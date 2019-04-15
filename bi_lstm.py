

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
print('hello')
# before proceeding further.
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import collections
import keras
import math
import numpy as np
import os
import random
import zipfile
from matplotlib import pylab
from six.moves import range
import tensorflow as tf
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
print('hello')
import pandas as pd
def createdata():
    pd.set_option('display.max_colwidth', -1)
    col = ['comments', 'label_object','label_binary']
    df_pos = pd.read_csv("posiive.csv", encoding='utf8', header=None, names=col)
    df_pos.drop_duplicates(subset="comments",
                           keep=False, inplace=True)
    df_pos.label_binary = 1
    # print('pos')
    # print(df_pos)
    col = ['comments', 'label_object','label_binary']
    df_neg = pd.read_csv("negative.csv", encoding='utf8', header=None, names=col)
    df_neg.label_binary = 0
    df_neg.drop_duplicates(subset="comments",
                         keep=False, inplace=True)

    frames = [df_pos, df_neg]
    result = pd.concat(frames)
    print(len(result))
    # print(result)
    return result,df_pos,df_neg
result,df_pos,df_neg = createdata()
result.dropna(inplace=True) #remove nan in dataset
#print(result.label_object)
x_comment=result.comments.tolist() #transfer series to list
y_labels_object = result.label_binary.tolist() #transfer series to list
def readdata():
    sentences = result.comments.tolist()  # transform a sentences to a list
    words = " ".join(sentences).split()  # transform list sentenses to list words
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())  # count frequent words appeard in list
    # Build dictionaries
    dic = dict()
    for word, _ in count:
        dic[word] = len(dic)  # dic, word -> id cats:0 dogs:1 ......
    reversed_dictionary = dict(zip(dic.values(), dic.keys()))
    voc_size = len(dic)
    # Make indexed word data
    data = list()  # list transform words to integer
    unk_count = 0
    for i in sentences:
        t = i.split()
        data1 = []
        for word in t:  # count rank for every word in words
            index = dic.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
            data1.append(index)
        data.append(data1)
    count[0][1] = unk_count
    data_recs = []
    return dic, reversed_dictionary, voc_size, data
dic,reversed_dictionary, voc_size, data = readdata()
max_length = max([len(s) for s in data]) #setting inputlength for embedding layer
embedding_size = 300 #hyper parameter for embedding size
x_data = np.asarray(data)
 #count y_labels
from keras.utils.np_utils import to_categorical
y_labels_object=to_categorical(y_labels_object) #create label
def splitdata(data,label): #create input data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels_object, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.4, random_state=42)
    X_valid = keras.preprocessing.sequence.pad_sequences(X_valid,
                                                        padding='post', maxlen=max_length)
    X_train = keras.preprocessing.sequence.pad_sequences(X_train,
                                                         padding='post', maxlen=max_length)
    X_test = keras.preprocessing.sequence.pad_sequences(X_test,
                                                        padding='post', maxlen=max_length)
    return X_train,X_test,X_valid,y_test,y_train,y_valid
X_train,X_test,X_valid,y_test,y_train,y_valid=splitdata(x_data,y_labels_object)
def model():
    embedding_layer = keras.layers.Embedding(voc_size, embedding_size)
    model = keras.Sequential()
    model.add(embedding_layer)
    #model.add(keras.layers.LSTM(units =64,activation='relu',return_sequences=True ))
    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(units=32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.03),
                          activity_regularizer=keras.regularizers.l2(0.03))))

    model.add(keras.layers.Dense(2, activation='softmax'))

    model.summary()
    my_optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=my_optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    earlyStopping = EarlyStopping(monitor='val_loss', verbose=1, mode='min',patience=3)
    mcp_save = ModelCheckpoint('model_file_bi-lstm.h5', save_best_only=True, monitor='val_acc', mode='max')
    history = model.fit(X_train,
                    y_train,
                    epochs=50,
                    batch_size=64,
                    validation_data=(X_valid,y_valid),
                    verbose=2,
                  callbacks=[earlyStopping, mcp_save])
    results = model.evaluate(X_test, y_test)
    print('this is result ',results)
    return history
history=model()
import h5py
#print(w)
def visualize(history):
    history_dict = history.history
    history_dict.keys()
    import matplotlib.pyplot as plt
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(acc) + 1)
    import csv
    with open('val_loss_1.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(map(lambda x: [x], val_loss))
    csvFile.close()
    with open('val_Acc_1.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(map(lambda x: [x], val_acc))
    csvFile.close()
    with open('loss_train_1.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(map(lambda x: [x], loss))
    csvFile.close()
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
visualize(history=history)
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        my_model = load_model('model_file_bi-lstm.h5')
y_pred = my_model.predict(X_test)
def predict():
    def newinput(sentences):
      # transform a sentences to a list
        # Make indexed word data
        data = list()  # list transform words to integer
        for i in sentences:
            t = i.split()
            data1 = []
            for word in t:  # count rank for every word in words
                index = dic.get(word, 0)
                data1.append(index)
            data.append(data1)
        return data
    with open('daura.csv', 'r', encoding='utf8') as file1:
        twt=[]
        for row in file1:
            twt.append(row)
    print(twt)
    twt_updated=newinput(twt)
    twt_updated=np.asarray(twt_updated)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt_updated = keras.preprocessing.sequence.pad_sequences(twt_updated,padding='post', maxlen=max_length)
    sentiment = my_model.predict_classes(twt_updated)
    #print('sentiment',sentiment)
    #print('y_test',y_test)
    print(sentiment)
predict()
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims = True)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Plot non-normalized confusion matrix
class_names = [0, 1]

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()