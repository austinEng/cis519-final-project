from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import time
import random
import collections
import numpy as np
import operator
import requests
from base64 import urlsafe_b64encode
# import tensorflow as tf
import tldextract

# from tensorflow.contrib import rnn

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

def readDictsFromFile():
  with open('browsingData.json') as json_data:
    data = json.load(json)
def loadData():
  key1 = '1xwUvmT4FPOyMReCUpmp'
  secret_key1 = '2ScKxRLSsy3FUX7GjAmh'
  key2 = 'tBIwxRy01HI6oLEjX03P'
  secret_key2 = '3l1UNkJsGIsqHe9j7MMi'
  key3 = 'TdVNTGXfnh8SL7C3yapI'
  secret_key3 = 'FYAjxschiCRZay6e4RF6'
  key4 = 'GDtuQ8ProJfzUGPv5nnR'
  secret_key4 = 'xnAyBDP35QQqh4kpcaGU'

  print("Loading data...")
  with open ('browsingData.json') as json_data:
    data = json.load(json_data)

  navigationData = data['navigationItems']
  historyData = [x for x in data['historyItems'] if 'id' in x]
  
  print("Processing data...")
  dictionary = dict()
  url_to_categories = dict()
  categories_dict = dict()
  idx = 1
  count = 0

  with open('category_data.json', 'a') as file:
    for historyItem in historyData:
      curr_url = historyItem['url']
      if "http" in curr_url[:5]:
        extracted_url = tldextract.extract(curr_url)
        shortened_url = ".".join(s.strip() for s in extracted_url if s.strip())

        if shortened_url not in url_to_categories:
          api_url = "https://api.webshrinker.com/categories/v3/{}".format(shortened_url)
          if count < 100:
            response = requests.get(api_url, auth=(key1, secret_key1))
          elif count < 200:
            response = requests.get(api_url, auth=(key2, secret_key2))
          elif count < 300:
            response = requests.get(api_url, auth=(key3, secret_key3))
          else:
            response = requests.get(api_url, auth=(key4, secret_key4))
          # try:
          status_code = response.status_code
          count += 1
          if status_code == 200:
            print (curr_url)
            try:
              data = response.json()
              # print ()
              # Do something with the JSON response
              category_data = data['data'][0]['categories']
              category_data.sort(key=operator.itemgetter('score'), reverse=True)
              file.write(json.dumps(category_data))
              categories_dict[category_data[0]['label']] = idx
              idx += 1
              url_to_categories[shortened_url] = category_data[0]['label']
            except:
              url_to_categories[shortened_url] = 'bad_url'
              categories_dict['bad_url'] = 0
          else:
            print (curr_url)
            print (status_code)
            url_to_categories[shortened_url] = 'bad_url'
            categories_dict['bad_url'] = 0
        if historyItem['url'] not in dictionary:
          dictionary[historyItem['url']] = categories_dict[url_to_categories[shortened_url]]

  with open('categories_dict.txt', 'w') as file:
    file.write(json.dumps(categories_dict))
  with open('url_to_categories.txt', 'w') as file:
    file.write(json.dumps(url_to_categories))
  with open('labels.txt', 'w') as file:
    file.write(json.dumps(dictionary))
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  label_count = len(categories_dict)

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
  loadData()
  # main()
