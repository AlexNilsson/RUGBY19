import os, math

import matplotlib.pyplot as plt

from keras.callbacks import Callback

class PlotLosses(Callback):
  def __init__(self):

    self.loss = []
    self.val_loss = []

    self.fig = plt.figure()
    plt.ion()
    plt.show()

    self.last_val_loss = 1000

  def on_epoch_end(self, epoch, logs={}):
    self.loss.append(logs.get('loss'))
    print(logs.get('val_loss'))
    if (logs.get('val_loss')):
      self.val_loss.append(logs.get('val_loss'))
      self.last_val_loss = logs.get('val_loss')
    else:
      self.val_loss.append(self.last_val_loss)

    x = [i+1 for i in range(len(self.loss))]

    plt.clf()
    plt.plot(x, self.loss, label="loss")
    plt.plot(x, self.val_loss, label="val_loss")
    plt.legend()
    plt.draw()
    plt.pause(0.001)

  #def on_batch_end(self, batch, logs={}):
    #self.on_epoch_end(batch, logs)
