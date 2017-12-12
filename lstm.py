from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import time
import random
import collections
import csv
import datetime

from numpy import newaxis
import sklearn.metrics
from matplotlib import pyplot


import numpy as np
import pandas as pd
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import  Sequential
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis

from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.contrib import rnn
import tldextract
look_back = 1
n_input = 50
n_hidden = 512
learning_rate = 0.001
training_iters = 50000
display_step = 100

def elapsed(sec):
  if sec<60:
    return str(sec) + " sec"
  elif sec<(60*60):
    return str(sec/60) + " min"
  else:
    return str(sec/(60*60)) + " hr"




def will_load_data(myfile = "converted_data.csv",interval_since_last_visited = 3600):
  print("Loading data...")
  set_of_domain_and_last_visited = {}
  # with open('convertedData', 'wb') as myfile:
  #   wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
  matrix = []
  map_of_domain_to_category = {}
  index = 0
  with open('browsingData.json') as json_data:
    data = json.load(json_data)
    historyData = [x for x in data['historyItems'] if 'id' in x]
    for historyItem in historyData:
      curr_url = historyItem['url']
      extracted_url = tldextract.extract(curr_url).domain
      if extracted_url not in map_of_domain_to_category:
        map_of_domain_to_category[extracted_url] = index
        index = index + 1


  with open('browsingData.json') as json_data:
    data = json.load(json_data)
    historyData = [x for x in data['historyItems'] if 'id' in x]
    for historyItem in historyData:
      current_input = []
      curr_url = historyItem['url']
      extracted_url = tldextract.extract(curr_url).domain
      time = historyItem['time']
      seconds = time / 1000.0

      d = datetime.datetime.fromtimestamp(seconds)
      hour_of_the_day = int(d.hour + d.minute / 60. + d.second / 3600)

      recently_visited = 0 if ((not extracted_url in set_of_domain_and_last_visited) or (seconds - set_of_domain_and_last_visited[extracted_url] > interval_since_last_visited)) else 1
      set_of_domain_and_last_visited[extracted_url] = seconds
      current_input.append(map_of_domain_to_category[extracted_url])
      current_input.append(hour_of_the_day)
      current_input.append(recently_visited)
      matrix.append(current_input)

  matrix = np.matrix(matrix)

  matrix = (matrix - np.mean(matrix, axis=1)) / np.std(matrix, axis=1)


  train_size = int(len(matrix) * 0.80)
  test_size = len(matrix) - train_size


  train = matrix[0:train_size]
  test = matrix[train_size:]
  print(train)

  return train, test







def loadData():
  print("Loading data...")
  with open ('browsingData.json') as json_data:
    data = json.load(json_data)
  set = {}
  navigationData = data['navigationItems']
  historyData = [x for x in data['historyItems'] if 'id' in x]


  print("Processing data...")
  dictionary = dict()
  idx = 0
  for historyItem in historyData:
    curr_url = historyItem['url']
    extracted_url = tldextract.extract(curr_url).domain
    if not extracted_url in set:
      set[extracted_url] = 1

    extracted_url = tldextract.extract(curr_url)
    print(historyItem)

    if historyItem['id'] not in dictionary:
      dictionary[historyItem['id']] = idx
      idx += 1

  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  label_count = len(dictionary)

  print("the sizew of set")
  print(len(set))
  for key, value in set.iteritems():
    print(key)
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


def predict_sequences_multiple(model, firstValue, length):
  prediction_seqs = []
  curr_frame = firstValue

  for i in range(length):
    predicted = []

    print(model.predict(curr_frame[newaxis, :, :]))
    predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])

    curr_frame = curr_frame[0:]
    curr_frame = np.insert(curr_frame[0:], i + 1, predicted[-1], axis=0)

    prediction_seqs.append(predicted[-1])

  return prediction_seqs

def create_dataset(dataset, look_back=1):
  dataX = []
  dataY = []

  for i in range(len(dataset)-look_back-1):
      a = dataset[i:(i+look_back)]
      if i == 2:
        print("LOOOl")
        print(a)
        print(dataset[i + look_back, 0])
      dataX.append(a)
      dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)

def main():
  start_time = time.time()
  train_data, test_data =  will_load_data()
  trainX, trainY = create_dataset(train_data, look_back)
  testX, testY = create_dataset(test_data, look_back)

  # trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
  # testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

  # Step 2 Build Model


  model = Sequential()
  model.add(LSTM(20, input_shape=(trainX.shape[1], trainX.shape[2])))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='rmsprop')
  history = model.fit(trainX, trainY, epochs=500, batch_size=20, validation_data=(testX, testY), verbose=2,
                      shuffle=False)
  pyplot.plot(history.history['loss'], label='train')
  pyplot.plot(history.history['val_loss'], label='test')
  pyplot.legend()
  pyplot.show()

  # make a prediction
  yhat = model.predict(testX)
  test_X = testX.reshape((testX.shape[0], testX.shape[2]))
  # invert scaling for forecast
  inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)

  scaler = MinMaxScaler(feature_range=(0, 1))


  inv_yhat = scaler.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:, 0]
  # invert scaling for actual
  test_y = testY.reshape((len(testY), 1))
  inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:, 0]
  # calculate RMSE
  rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
  print('Test RMSE: %.3f' % rmse)




if __name__ == "__main__":
  main()
