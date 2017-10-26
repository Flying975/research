#!/usr/bin/env python

"""
Usage:
>> ./server.py
>> ./train_generator.py autoencoder
"""
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import argparse
import time
from keras import callbacks as cbks
import logging
import tensorflow as tf
import numpy as np

from server import client_generator
from models.utils import save_images
mixtures = 1
import cv2
import matplotlib.pyplot as plt
import glob
def old_cleanup(data):
  X = data[0]
  if X.shape[1] == 1:
    X = X[:, -1, :]/127.5 - 1.
  return X


def gen(hwm, host, port):
  for tup in client_generator(hwm=hwm, host=host, port=port):
    X = cleanup(tup)
    yield X


def train_model(name, g_train, d_train, sampler, generator, samples_per_epoch, nb_epoch,
                z_dim=100, verbose=1, callbacks=[],
                validation_data=None, nb_val_samples=None,
                saver=None,sess=None, extras =None):
    """
    Main training loop.
    modified from Keras fit_generator
    """
    self = {}
    epoch = 0
    counter = 0
    out_labels = ['g_loss', 'd_loss', 'd_loss_fake', 'd_loss_legit', 'time']  # self.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]
    G, D, E = extras
    file_path = "./dataset/int_data"


    # prepare callbacks
    history = cbks.History()
    callbacks = [cbks.BaseLogger()] + callbacks + [history]
    if verbose:
        callbacks += [cbks.ProgbarLogger()]
    callbacks = cbks.CallbackList(callbacks)

    callbacks.set_params({
        'epochs': nb_epoch,
        'samples': samples_per_epoch,
        'verbose': verbose,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    while epoch < nb_epoch:

      callbacks.on_epoch_begin(epoch)
      samples_seen = 0
      batch_index = 0
      while samples_seen < samples_per_epoch:
        gen_interp_img(epoch, E, G, file_path)
        z, x = next(generator)
        # build batch logs
        batch_logs = {}
        if type(x) is list:
          batch_size = len(x[0])
        elif type(x) is dict:
          batch_size = len(list(x.values())[0])
        else:
          batch_size = len(x)
        batch_logs['batch'] = batch_index
        batch_logs['size'] = batch_size
        callbacks.on_batch_begin(batch_index, batch_logs)

        t1 = time.time()
        d_losses = d_train(x, z, counter)
        z, x = next(generator)
        g_loss, samples, xs = g_train(x, z, counter)
        outs = (g_loss, ) + d_losses + (time.time() - t1, )
        counter += 1

        # save samples

        if batch_index % 100 == 0:
          join_image = np.zeros_like(np.concatenate([samples[:64], xs[:64]], axis=0))
          for j, (i1, i2) in enumerate(zip(samples[:64], xs[:64])):
            join_image[j*2] = i1
            join_image[j*2+1] = i2
          save_images(join_image, [8*2, 8],
                      './outputs/samples_%s/train_%s_%s.png' % (name, epoch, batch_index))

          samples, xs = sampler(z, x)
          join_image = np.zeros_like(np.concatenate([samples[:64], xs[:64]], axis=0))
          for j, (i1, i2) in enumerate(zip(samples[:64], xs[:64])):
            join_image[j*2] = i1
            join_image[j*2+1] = i2
          save_images(join_image, [8*2, 8],
                      './outputs/samples_%s/test_%s_%s.png' % (name, epoch, batch_index))

        for l, o in zip(out_labels, outs):
            batch_logs[l] = o

        callbacks.on_batch_end(batch_index, batch_logs)

        # construct epoch logs
        epoch_logs = {}
        batch_index += 1
        samples_seen += batch_size

      if saver is not None:
        saver(epoch)

      callbacks.on_epoch_end(epoch, epoch_logs)
      epoch += 1

    # _stop.set()
    callbacks.on_train_end()

def prepare__data(path):
    file_list = glob.glob('{}/*.jpg'.format(path))
    imgs = [cv2.resize(cv2.imread(i),(160,80)) for i in file_list]

    #imgs = [plt.imread(i) for i in file_list]
    imgs = np.asarray(imgs,np.float32)
    print("imgs.shape===", imgs.shape)
    imgs = imgs/127.5 -1.
    return imgs


def get_img_code(imgs, E):
    print('inter ,x',imgs[0])
    batch_size = imgs.shape[0]
    #print("batch_size=", batch_size)
    #print("imgs.shape===", imgs.shape)
    encode_out = E.predict(imgs,batch_size = batch_size)
    print('inter zout\n',encode_out[0])
    shape = encode_out[0].shape

    noise = np.random.normal(0., 1., shape)
    codes = encode_out[0] + noise * encode_out[1]
    z_inter = get_interpolation(codes, 8, shape[-1])
    print('inter z',z_inter[0])


    return z_inter
def get_interpolation(codes,N, z_dim):
    shape = codes.shape
    code_a = codes[:int(shape[0]/2)]
    code_a = np.reshape(code_a,[-1, 1, z_dim])
    code_b = codes[int(shape[0]/2):]
    code_b = np.reshape(code_b, [-1, 1, z_dim])
    alpha = np.reshape(np.linspace(0., 1., N), [1, N, 1])
    z_inter = alpha*code_a + (1 - alpha)*code_b
    z_inter = np.reshape(z_inter , [-1 , z_dim])
    # print('interp  z \n', z_inter)

    return z_inter

def gen_interp_img(epoch ,E,G,file_path ,output_path = None):
    imgs = prepare__data(file_path)
    z_inter = get_img_code(imgs, E)

    decode_img = G.predict(z_inter)

    save_imgs(decode_img, epoch)

def save_imgs(decode_img , epoch):
    big_img = np.ones([640, 1600, 3], np.float32)
    print('iter code img \n',decode_img[0])

    o_shape = [80,160]
    for i in range(8):
        for j in range(8):
            big_img[i * o_shape[0]:(i + 1) * o_shape[0], j * o_shape[1]:(j + 1) * o_shape[1]] = decode_img[i * 8 + j][:]

    print('big_img \n',big_img)
    big_img = (big_img + 1.)*255./2.

    cv2.imwrite('./int/{}.jpg'.format(epoch),big_img )

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generative model trainer')
  parser.add_argument('model', type=str, default="bn_model", help='Model definitnion file')
  parser.add_argument('--name', type=str, default="autoencoder", help='Name of the model.')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  # parser.add_argument('--time', type=int, default=1, help='How many temporal frames in a single input.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=1, help='Number of epochs.')
  parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
  parser.add_argument('--epochsize', type=int, default=100, help='How many frames per epoch.')
  parser.add_argument('--loadweights', dest='loadweights', action='store_true', help='Start from checkpoint.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=True)
  args = parser.parse_args()

  MODEL_NAME = args.model
  logging.info("Importing get_model from {}".format(args.model))
  exec("from models."+MODEL_NAME+" import get_model")
  # try to import `cleanup` from model file
  try:
    exec("from models."+MODEL_NAME+" import cleanup")
  except:
    cleanup = old_cleanup

  model_code = open('models/'+MODEL_NAME+'.py').read()

  if not os.path.exists("./outputs/results_"+args.name):
      os.makedirs("./outputs/results_"+args.name)
  if not os.path.exists("./outputs/samples_"+args.name):
      os.makedirs("./outputs/samples_"+args.name)
  #fix OOM
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with tf.Session(config=config) as sess:
    g_train, d_train, sampler, saver, loader, extras = get_model(sess=sess, name=args.name, batch_size= None, gpu=args.gpu)

    # start from checkpoint
    if args.loadweights:
      print('loading weight')
      #sess.run(tf.global_variables_initializer())
      loader()
    else:
      sess.run(tf.global_variables_initializer())


    train_model(args.name, g_train, d_train, sampler,
                gen(20, args.host, port=args.port),
                samples_per_epoch=args.epochsize,
                nb_epoch=args.epoch, verbose=1, saver=saver,
                sess=sess, extras = extras)
