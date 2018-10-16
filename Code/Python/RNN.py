
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import os
import sys
import numpy as np
np.set_printoptions(threshold=np.nan)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Starting up")

from readData import EEG
eeg = EEG()
eeg.input_data("EEG_data") #

# Training Parameters
learning_rate = 0.001
training_epochs = 100
early_stop_counter = 0
n_early_stop_epochs = 5
batch_size = 64
display_step = 36

# Network Parameters
num_input = 14 # MNIST data input (img shape: 28*28)
timesteps = 1 # timesteps
num_hidden = 256 # hidden layer num of features
num_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    min_v_loss = sys.float_info.max

    # Run the initializer
    sess.run(init)

    for epoch in range(training_epochs):
        eeg.reset_batch_index()

        training_steps = int(eeg.X_train_list.shape[0]/batch_size)
        avg_loss = 0.
        avg_acc = 0.

        for step in range(1, training_steps+1):
            batch_x, batch_y = eeg.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                    Y: batch_y})
            
            avg_loss += loss / training_steps
            avg_acc += acc / training_steps

        v_loss, v_acc = sess.run([loss_op, accuracy], feed_dict={X: eeg.X_valid_list.reshape((-1, timesteps, num_input)),
                                                            Y: eeg.Y_valid_list})

        print("Epoch:", '%04d' % (epoch+1), "loss={:.9f}".format(avg_loss), "accuracy={:.9f}".format(avg_acc))
        print("validation loss={:.9f}".format(v_loss), "validation accuracy={:.9f}".format(v_acc))

        if v_loss < min_v_loss: 
            print('new best epoch: ', '%04d' % (epoch+1))
            min_v_loss = v_loss 
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter > n_early_stop_epochs:
            # too many consecutive epochs without surpassing the best model
            print('stopping early')
            break
        
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 64
    test_data = eeg.X_test_list.reshape((-1, timesteps, num_input))
    test_label = eeg.Y_test_list
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
