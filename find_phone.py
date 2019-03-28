{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from skimage import data\n",
    "import matplotlib.pyplot as plt\n",
    "import random\n",
    "import os\n",
    "import sys\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def acc(y,pred,err):\n",
    "  #return np.sum(np.linalg.norm((y-pred),axis=1)<err)\n",
    "  norm = tf.norm(y - pred, axis=1)\n",
    "  acc = tf.less_equal(norm, err)\n",
    "  return tf.reduce_sum(tf.cast(acc, tf.float32))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "imgpath = sys.argv[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tf.reset_default_graph()\n",
    "\n",
    "x = tf.placeholder(dtype = tf.float32, shape=[None,326,490,3])\n",
    "y = tf.placeholder(dtype=tf.float32,shape=[None,2])\n",
    "is_train = tf.placeholder(dtype=tf.bool, shape=())\n",
    "\n",
    "#images_flatten = tf.contrib.layers.flatten(x)\n",
    "\n",
    "conv1 = tf.nn.conv2d(x, tf.get_variable('w_1', [11,11,3,6], tf.float32, tf.random_normal_initializer(0.0, 0.02))\n",
    "                     , [1, 2, 2, 1], padding='SAME') + tf.get_variable('b_1', [1,1,1,6], initializer=tf.constant_initializer(0.0))\n",
    "norm1 = tf.layers.batch_normalization(conv1)\n",
    "pool1 = tf.nn.max_pool(norm1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')\n",
    "relu1 = tf.nn.leaky_relu(pool1)\n",
    "\n",
    "\n",
    "conv2 = tf.nn.conv2d(relu1, tf.get_variable('w_2', [5,5,6,6], tf.float32, tf.random_normal_initializer(0.0, 0.02))\n",
    "                     , [1, 1, 1, 1], padding='SAME') + tf.get_variable('b_2', [1,1,1,6], initializer=tf.constant_initializer(0.0))\n",
    "norm2 = tf.layers.batch_normalization(conv2)\n",
    "pool2 = tf.nn.max_pool(norm2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')\n",
    "relu2 = tf.nn.leaky_relu(pool2)\n",
    "\n",
    "\n",
    "\n",
    "flat = tf.contrib.layers.flatten(relu2)\n",
    "fc1 = tf.contrib.layers.fully_connected(flat, 2048 , tf.nn.leaky_relu)\n",
    "fc2 = tf.contrib.layers.fully_connected(fc1, 2048, tf.nn.leaky_relu)\n",
    "fc3 = tf.contrib.layers.fully_connected(fc2, 512, tf.nn.leaky_relu)\n",
    "fc4 = tf.contrib.layers.fully_connected(fc3, 128, tf.nn.leaky_relu)\n",
    "fc5 = tf.contrib.layers.fully_connected(fc4, 2 ,activation_fn=None)\n",
    "out = tf.nn.sigmoid(fc5)\n",
    "# out = fc3\n",
    "#loss function\n",
    "#loss = tf.reduce_mean(tf.nn.l2_loss(y - fc2))\n",
    "loss = tf.divide(tf.nn.l2_loss(y - out), tf.cast(tf.shape(y)[0], tf.float32))\n",
    "#optimizer\n",
    "global_step = tf.Variable(0, trainable=False)\n",
    "starter_learning_rate = 0.00001\n",
    "# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,\n",
    "#                                            num_batches*10, 0.96, staircase=True)\n",
    "\n",
    "\n",
    "update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)\n",
    "train_op = tf.train.AdamOptimizer(learning_rate=starter_learning_rate).minimize(loss)\n",
    "train_op = tf.group([train_op, update_ops])\n",
    "\n",
    "accuracy = acc(y, out, 0.05)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "sess = tf.Session()\n",
    "sess.run(tf.global_variables_initializer())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "variables = tf.global_variables()\n",
    "param_dict = {}\n",
    "for var in variables:\n",
    "    var_name = var.name[:-2]\n",
    "    print('Loading {} from checkpoint. Name: {}'.format(var.name, var_name))\n",
    "    param_dict[var_name] = var\n",
    "saver = tf.train.Saver()\n",
    "saver.restore(sess, \"trained_model.ckpt\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "img = cv2.imread(imgpath)\n",
    "w,h,c=img.shape\n",
    "img = img.reshape(1,w,h,c)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def output(y):\n",
    "  return y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "rand_label=(np.random.rand(),np.random.rand())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "output_coordinate = sess.run([output],\n",
    "                    feed_dict = {\n",
    "                        x: img,\n",
    "                        y: rand_label,\n",
    "                        is_train:False\n",
    "                    })\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(output_coordinate)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
