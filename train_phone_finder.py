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
    "import cv2\n",
    "import sys"
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
    "def shuffle(a, b):\n",
    "  state = np.random.get_state()\n",
    "  np.random.shuffle(a)\n",
    "  np.random.set_state(state)\n",
    "  np.random.shuffle(b)\n",
    "  return a, b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "path = sys.argv[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "images = []\n",
    "labels = []\n",
    "lines = {}\n",
    "\n",
    "with open(path+\"labels.txt\",\"r\") as file:\n",
    "  for line in file:\n",
    "    a=line.split()\n",
    "    lines[a[0]] = np.array(( float(a[1]), float(a[2]) ))\n",
    "\n",
     # Create sample images and labels by flipping horizontally, vertically and horizontal+vertically
    "for f in os.listdir(path):\n",
    "  if f.endswith('.jpg'):\n",
    "    labels.append(lines[f])\n",
    "    img = cv2.imread(path+f)\n",
    "    bright_img = cv2.add(img,10)\n",
    "    images.append(img)\n",
    "    images.append(bright_img)\n",
    "    labels.append(lines[f])\n",
    "    flip = cv2.flip(img,flipCode=0)\n",
    "    labels.append(np.array((lines[f][0],1-lines[f][1])))\n",
    "    images.append(flip)\n",
    "    flip = cv2.flip(img,flipCode=1)\n",
    "    labels.append(np.array((1-lines[f][0],lines[f][1])))\n",
    "    images.append(flip)\n",
    "    flip = cv2.flip(flip,flipCode=0)\n",
    "    labels.append(np.array((1-lines[f][0],1-lines[f][1])))\n",
    "    images.append(flip)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "img1 = np.array((images))\n",
    "#print(img1[0])\n",
    "mean = np.mean(img1,axis=0)\n",
    "#print(mean)\n",
    "img1 = img1-mean\n",
    "#print(img1[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('mean.txt','w') as f:\n",
    "  f.write(str(mean))\n",
    "f.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size =50\n",
    "n = len(img1)\n",
    "epochs = 20"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "img1, labels = shuffle(img1, labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_train = n\n",
    "num_batches = n_train//batch_size + 1"
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
    "tf.set_random_seed(500)\n",
    "\n",
    "sess = tf.Session()\n",
    "sess.run(tf.global_variables_initializer())\n",
    "\n",
    "best_accu_val = 0.0\n",
    "best_weights = {}\n",
    "\n",
    "for i in range(20):\n",
    "  for j in range(num_batches):\n",
    "    # Train batch\n",
    "    _, loss_, accu_, y_, out_ = sess.run([train_op,loss,accuracy, y, out],\n",
    "                                         feed_dict = {\n",
    "                                             x: img1[j*batch_size:min((j+1)*batch_size, n_train)],\n",
    "                                             y: labels[j*batch_size:min((j+1)*batch_size, n_train)],\n",
    "                                             is_train: True\n",
    "                                         })\n",
    "    # Validation\n",
    "    loss_val, accu_val = sess.run([loss,accuracy],\n",
    "                                         feed_dict = {\n",
    "                                             x: img1[-20:],\n",
    "                                             y: labels[-20:],\n",
    "                                             is_train: False\n",
    "                                         })\n",
    "    \n",
    "    if(accu_val >= best_accu_val or best_weights is None):\n",
    "      best_accu_val = accu_val\n",
    "      variables_names =[v.name for v in tf.trainable_variables()]\n",
    "      values = sess.run(variables_names)\n",
    "      for k,v in zip(variables_names, values):\n",
    "        best_weights[k] = v\n",
    "        \n",
    "        \n",
    "    print(\"Epoch : {}, iteration : {}, train loss : {:.5f}, train accuracy:{}, val loss : {:.5f}, val accuracy: {}\".format(i, j, loss_,accu_, loss_val, accu_val))\n",
    "#   print(fc1_.shape)\n",
    "#   print(fc2_.shape)\n",
    "#   print(fc3_.shape)\n",
    "#   print(fc3_)\n",
    "#   print(out_.shape)\n",
    "#   print(out_)\n",
    "#   print(y_)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "saver = tf.train.Saver()\n",
    "model_path = saver.save(sess, \"trained_model.ckpt\")"
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
