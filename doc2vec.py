
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
print('hello')
import matplotlib.pyplot as plt
# before proceeding further.
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
import csv
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
result.dropna(inplace=True)
#print(result.label_object)
tongodai=len(result.label_object)
x_comment=result.comments.tolist()
y_labels_object = result.label_object.tolist()
y_label_binary=result.label_binary.tolist()
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
def savedic():
    w = csv.writer(open("dic.csv", "w"))
    for key, val in dic.items():
        w.writerow([key, val])
savedic()
max_length = max([len(s) for s in data])
embedding_size = 100
x_data = np.asarray(data)
from keras.utils.np_utils import to_categorical
y_label_binary=to_categorical(y_label_binary)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_label_binary, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.4, random_state=42)
X_valid = keras.preprocessing.sequence.pad_sequences(X_valid,
                                                    padding='post', maxlen=max_length)
X_train = keras.preprocessing.sequence.pad_sequences(X_train,
                                                     padding='post', maxlen=max_length)
X_test = keras.preprocessing.sequence.pad_sequences(X_test,
                                                    padding='post', maxlen=max_length)
from keras.layers import LeakyReLU,ReLU,PReLU
def model():
    my_optimizer = keras.optimizers.Adam(lr=0.0001)
    embedding_layer = keras.layers.Embedding(voc_size, embedding_size,input_length=max_length)
    model = keras.Sequential()
    model.add(embedding_layer)
    model.add(keras.layers.GlobalMaxPool1D())
    model.add(keras.layers.Dense(8, kernel_initializer='normal', kernel_regularizer=keras.regularizers.l2(0.03),
                                 activity_regularizer=keras.regularizers.l2(0.04)))
    model.add(keras.layers.BatchNormalization())
    model.add(ReLU())
    model.add(keras.layers.Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=my_optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    earlyStopping = EarlyStopping(monitor='val_loss', verbose=1, mode='min', patience=2)
    mcp_save = ModelCheckpoint('model_file_doc2vec.h5', save_best_only=True, monitor='val_acc', mode='max')
    history = model.fit(X_train,
                        y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_valid, y_valid),
                        verbose=2,
                        callbacks=[earlyStopping, mcp_save])
    results = model.evaluate(X_test, y_test)
    print('this is result ',results)
    return history,model
history,trained_model=model()
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
from sklearn.metrics import confusion_matrix
y_pred=trained_model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test=y_test.argmax(axis=1)
cnf_matrix = confusion_matrix(y_test, y_pred)
normalized_confusion_matrix = cnf_matrix/cnf_matrix.sum(axis = 1, keepdims = True)
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
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
from sklearn import metrics
score=metrics.f1_score(y_test, y_pred)
print(score)
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)