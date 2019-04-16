#from _future_ import print_function
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import pandas as pd
def createdata():
    pd.set_option('display.max_colwidth', -1)
    col=['comments','label']
    df_pos = pd.read_csv("Output_Pos4.csv",encoding='utf8',header=None,names=col)
    df_pos.label=1
    col=['comments','label']
    df_neg = pd.read_csv("Output_Neg4.csv",encoding='utf8',header=None,names=col)
    df_neg.label=0
    frames = [df_pos, df_neg]
    result = pd.concat(frames)
    return result
result=createdata()
#def create_training_data(corpus_raw):
print('batdau')
sentences = result["comments"].tolist()
print('sentences',sentences)
normalized_sentences = []
for sentence in sentences:
    normalized_sentences.append(sentence)
# sentences to words and count
print('normalized sentencses',normalized_sentences)
words = " ".join(normalized_sentences).split()
print('point0')
print('words-------------------------',words)
print('point1')
count = [['UNK', -1]]
count.extend(collections.Counter(words).most_common())
print('count',count)
dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary)
print('dict',dictionary)
data = list()
unk_count = 0
for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
        unk_count += 1
    data.append(index)
print('data',data[0])
count[0][1] = unk_count
reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
print('reserved_dic',reversed_dictionary[0])
"""
return data, count, dictionary, reversed_dictionary
data, count, unused_dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    words_list_trained = []
    words_list=[]
    for sent in corpus_raw.split('.'):
        for w in sent.split():
            if w != '.':
                words_list_trained.append(w.split('.')[0])
        words_list_trained.append('<end>')
    print('day la words_list_trained',words_list_trained)
    for sent in corpus_raw.split('.'):
        for w in sent.split():
            if w != '.':
                words_list.append(w.split('.')[0])
    print('day la words_list',words_list)
    count = [['UNK', -1]]
    vocabulary_size=len(list(set(words_list)))
    print(vocabulary_size)
    count.extend(collections.Counter(words_list).most_common(vocabulary_size - 1))
    print('day la count',count)
    print('len count',len(count))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    print('day la dictionary',dictionary)
    data = list()
    unk_count = 0
    for word in words_list:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    print(count)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print('reserver dic',reversed_dictionary)
    return data, count, dictionary, reversed_dictionary,vocabulary_size
corpus_raw = (corpus_raw).lower()
data, count, dictionary,reversed_dictionary,vocabulary_size=create_training_data(corpus_raw)
data_index = 0
print('Most common words (+UNK)', count[:5])
print(data)
print([reversed_dictionary[i] for i in data])
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
      context_words = [w for w in range(span) if w != skip_window]
      words_to_use = random.sample(context_words, num_skips)
      for j, context_word in enumerate(words_to_use):
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[context_word]
      if data_index == len(data):
        #print('het 1 epoch---------------------------------------------------')
        buffer.extend(data[0:span])
        data_index = span
      else:
        buffer.append(data[data_index])
        data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels
batch, labels = generate_batch(batch_size=64, num_skips=2, skip_window=1)
for i in range(64):
    print(batch[i], reversed_dictionary[batch[i]], '->', labels[i, 0],
          reversed_dictionary[labels[i, 0]])
batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.
graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                   labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's minimize method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on tf.train.Optimizer.minimize() for more details.
    optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(loss)
    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
num_steps = 30000
print('day la len reserver',len(reversed_dictionary))
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  #print('bandau', embed.eval())
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reversed_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        #print(nearest)
        #print(nearest[0])
        #print('day la nearest',reversed_dictionary[nearest[0]])
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reversed_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        #print(log)
  final_embeddings = normalized_embeddings.eval()
num_points = 100
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                  ha='right', va='bottom')
  pylab.show()
words = [reversed_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
"""