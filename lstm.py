from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import random
import collections
from urlparse import urlparse
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

n_input = 10
n_hidden = 200
timeThreshold = 30
learning_rate = 0.001
training_iters = 500000
display_step = 100

def elapsed(sec):
  if sec<60:
    return str(sec) + " sec"
  elif sec<(60*60):
    return str(sec/60) + " min"
  else:
    return str(sec/(60*60)) + " hr"

def toHours(ms):
  d = datetime.fromtimestamp(ms / 1000.0)
  return d.hour + d.minute / 60.0 + d.second / 3600.0

def loadData():
  print("Loading data...")
  with open ('browsingData.json') as json_data:
    data = json.load(json_data)

  navigationData = data['navigationItems']
  historyData = [x for x in data['historyItems'] if 'id' in x]

  print(len(navigationData))
  print(len(historyData))

  print("Processing data...")
  idDictionary = dict()
  netlocDictionary = dict()
  idx = 0
  netlocIdx = 0
  for i, historyItem in enumerate(historyData):
    # ignore last item because we diff sequential items
    if i + 1 >= len(historyData):
      break

    historyItem['lastVisitTimeHours'] = toHours(historyItem['lastVisitTime'])
    historyItem['timeHours'] = toHours(historyItem['time'])
    historyItem['duration'] = (historyData[i + 1]['time'] - historyItem['time']) / 1000.0

    if historyItem['id'] not in idDictionary:
      idDictionary[historyItem['id']] = idx
      idx += 1

    historyItem['netloc'] = urlparse(historyItem['url']).netloc
    if historyItem['netloc'] not in netlocDictionary:
      netlocDictionary[historyItem['netloc']] = netlocIdx
      netlocIdx += 1

  historyData.pop()
  historyData = [d for d in historyData if d['duration'] >= timeThreshold]
  print(len(historyData), 'filter items')

  idReverseDictionary = dict(zip(idDictionary.values(), idDictionary.keys()))
  netlocReverseDictionary = dict(zip(netlocDictionary.values(), netlocDictionary.keys()))

  return {
    'historyData': historyData,
    'idDictionary': idDictionary,
    'idReverseDictionary': idReverseDictionary,
    'idCount': len(idDictionary),
    'netlocDictionary': netlocDictionary,
    'netlocReverseDictionary': netlocReverseDictionary,
    'netlocCount': len(netlocDictionary)
  }


def RNN(x, weights, biases):
  x = tf.reshape(x, [-1, n_input])
  x = tf.split(x, n_input, 1)

  rnn_cell = rnn.MultiRNNCell([
    # rnn.LayerNormBasicLSTMCell(n_hidden),
    # rnn.LayerNormBasicLSTMCell(n_hidden),
    # rnn.LSTMCell(n_hidden),
    # rnn.LSTMCell(n_hidden),
    # rnn.DropoutWrapper(rnn.LSTMCell(n_hidden), input_keep_prob=0.9),
    # rnn.DropoutWrapper(rnn.LSTMCell(n_hidden), input_keep_prob=0.9)
    # rnn.DropoutWrapper(rnn.GRUCell(n_hidden), input_keep_prob=0.9),
    # rnn.DropoutWrapper(rnn.GRUCell(n_hidden), input_keep_prob=0.9)
    rnn.DropoutWrapper(rnn.LayerNormBasicLSTMCell(n_hidden), input_keep_prob=0.9),
    rnn.DropoutWrapper(rnn.LayerNormBasicLSTMCell(n_hidden), input_keep_prob=0.9),
  ], state_is_tuple=True)

  # rnn_cell = rnn.BasicLSTMCell(n_hidden)

  outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

  return tf.matmul(outputs[-1], weights['out']) + biases['out']


def main():
  start_time = time.time()

  data = loadData()
  # historyData = data['historyData']
  # historyData, dictionary, reverse_dictionary, label_count = loadData()
  label_count = data['netlocCount']
  print(label_count, 'labels')

  x = tf.placeholder("float", [None, n_input, 1])
  y = tf.placeholder("float", [None, label_count])

  weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, label_count]))
  }

  biases = {
    'out': tf.Variable(tf.random_normal([label_count]))
  }

  pred = RNN(x, weights, biases)

  # Loss and optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred[0], labels=y))
  optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
  # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

  # Model evaluation
  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Initializing the variables
  init = tf.global_variables_initializer()

  logs_path = '/tmp/tensorflow/rnn_browsing'
  writer = tf.summary.FileWriter(logs_path)

  # Launch the graph
  with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    running_accuracy = 0

    writer.add_graph(session.graph)

    print("Training model...")
    while step < training_iters:
      if offset > len(data['historyData']) - end_offset:
        offset = random.randint(0, n_input+1)

      inputs = [
        [
          # title
          # url
          data['netlocDictionary'][data['historyData'][i]['netloc']],
          data['idDictionary'][data['historyData'][i]['id']],
          data['historyData'][i]['typedCount'],
          data['historyData'][i]['lastVisitTime'],
          data['historyData'][i]['lastVisitTimeHours'],
          data['historyData'][i]['visitCount'],
          data['historyData'][i]['time'],
          data['historyData'][i]['timeHours'],
          data['historyData'][i]['duration'],
          (data['historyData'][offset + n_input]['time'] - data['historyData'][i]['time']) / 1000.0,
        ] for i in range(offset, offset + n_input)
      ]
      # inputs.append([ historyData[offset + n_input]['time'] ])
      inputs = np.reshape(np.array(inputs), [-1, n_input, 1])

      label = data['historyData'][offset + n_input]['netloc']
      onehot_out = np.zeros([label_count], dtype=float)
      onehot_out[data['netlocDictionary'][label]] = 1.0
      onehot_out = np.reshape(onehot_out,[1,-1])

      _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={x: inputs, y: onehot_out})

      # predicted_label = data['netlocReverseDictionary'][int(tf.argmax(onehot_pred, 1).eval()[0])]
      # print('Predicted: {:<50} Actual: {:<50}'.format(predicted_label, label))
      running_accuracy = (1 - learning_rate) * running_accuracy + learning_rate * acc

      loss_total += loss
      acc_total += acc

      if (step+1) % display_step == 0:
        print("Iter= " + str(step+1) + ", Average Loss= " + \
              "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
              "{:.2f}%".format(100*acc_total/display_step) + ", Running Accuracy= " + \
              "{:.2f}%".format(100*running_accuracy))
        acc_total = 0
        loss_total = 0
        # symbols_in = [historyData[i] for i in range(offset, offset + n_input)]
        # symbols_out = historyData[offset + n_input]
        # symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
        # print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))

      step += 1
      offset += (n_input+1)

    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))


if __name__ == "__main__":
  main()
