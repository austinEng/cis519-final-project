from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import random
import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

n_input = 50
n_hidden = 512
learning_rate = 0.001
training_iters = 50000
display_step = 10

def elapsed(sec):
  if sec<60:
    return str(sec) + " sec"
  elif sec<(60*60):
    return str(sec/60) + " min"
  else:
    return str(sec/(60*60)) + " hr"

def loadData():
  print("Loading data...")
  with open ('browsingData.json') as json_data:
    data = json.load(json_data)

  navigationData = data['navigationItems']
  historyData = [x for x in data['historyItems'] if 'id' in x]

  print("Processing data...")
  dictionary = dict()
  idx = 0
  for historyItem in historyData:
    if historyItem['id'] not in dictionary:
      dictionary[historyItem['id']] = idx
      idx += 1

  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  label_count = len(dictionary)

  for v in dictionary.values():
    if v >= label_count:
      raise ValueError

  return (historyData, dictionary, reverse_dictionary, label_count)

def RNN(x, weights, biases):
  x = tf.reshape(x, [-1, n_input])
  x = tf.split(x, n_input, 1)

  rnn_cell = rnn.MultiRNNCell([
    rnn.BasicLSTMCell(n_hidden),
    rnn.BasicLSTMCell(n_hidden),
  ])

  # rnn_cell = rnn.BasicLSTMCell(n_hidden)

  outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

  return tf.matmul(outputs[-1], weights['out']) + biases['out']

def main():
  start_time = time.time()

  historyData, dictionary, reverse_dictionary, label_count = loadData()

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

    writer.add_graph(session.graph)

    print("Training model...")
    while step < training_iters:
      if offset > len(historyData) - end_offset:
        offset = random.randint(0, n_input+1)

      inputs = [
        [
          # title
          # url
          dictionary[historyData[i]['id']],
          historyData[i]['typedCount'],
          historyData[i]['lastVisitTime'],
          historyData[i]['visitCount'],
          historyData[i]['time'],
        ] for i in range(offset, offset + n_input)
      ]
      # inputs.append([ historyData[offset + n_input]['time'] ])
      inputs = np.reshape(np.array(inputs), [-1, n_input, 1])

      label = historyData[offset + n_input]['id']
      onehot_out = np.zeros([label_count], dtype=float)
      onehot_out[dictionary[label]] = 1.0
      onehot_out = np.reshape(onehot_out,[1,-1])

      _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], feed_dict={x: inputs, y: onehot_out})

      # predicted_label = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval()[0])]
      # print('Predicted:', predicted_label, 'Actual:', label)

      loss_total += loss
      acc_total += acc

      if (step+1) % display_step == 0:
        print("Iter= " + str(step+1) + ", Average Loss= " + \
              "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
              "{:.2f}%".format(100*acc_total/display_step))
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
