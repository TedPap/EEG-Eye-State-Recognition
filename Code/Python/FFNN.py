from __future__ import print_function
import os
import sys
import numpy as np
np.set_printoptions(threshold=np.nan)
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Starting up")

from readData import EEG
eeg = EEG()
eeg.input_data("EEG_data") #


# Parameters
learning_rate = 0.001
training_epochs = 200
early_stop_counter = 0
n_early_stop_epochs = 100
batch_size = 64
display_step = 10

# Network Parameters
n_hidden_1 = 256 # 1st 256 layer number of neurons
n_hidden_2 = 256 # 2nd 256 layer number of neurons
n_input = 14
n_classes = 2
dropout = 0.8 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def compute_log(X):
    X[ X <= 0 ] = 0.01
    return np.log(X)


# Create model
def MLP(x, dropout):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, dropout)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Construct model
logits = MLP(X, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:

    min_v_loss = sys.float_info.max

    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):

        eeg.reset_batch_index()
        eeg.reshuffle(eeg.X_train_list, eeg.Y_train_list)
        eeg.reshuffle(eeg.X_test_list, eeg.Y_test_list)
        eeg.reshuffle(eeg.X_valid_list, eeg.Y_valid_list)

        avg_loss = 0.
        avg_acc = 0.
        total_batch = int(eeg.X_train_list.shape[0]/batch_size)

        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = eeg.next_batch(batch_size)

            # batch_x = compute_log(batch_x)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, loss, acc = sess.run([train_op, loss_op, accuracy], feed_dict={X: batch_x,
                                                            Y: batch_y,
                                                            keep_prob: 0.8})
            # Compute average loss
            avg_loss += loss / total_batch
            avg_acc += acc / total_batch

            # valid_batch = compute_log(eeg.X_valid_list)
            valid_batch = eeg.X_valid_list

        v_loss, v_acc = sess.run([loss_op, accuracy], feed_dict={X: eeg.X_valid_list,
                                                            Y: eeg.Y_valid_list,
                                                            keep_prob: 1.0})

        if v_loss < min_v_loss: 
            print('new best epoch: ', '%04d' % (epoch+1))
            min_v_loss = v_loss 
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss={:.9f}".format(avg_loss), "accuracy={:.9f}".format(avg_acc))
            print("validation loss={:.9f}".format(v_loss), "validation accuracy={:.9f}".format(v_acc))
            print(" ")
        
        if early_stop_counter > n_early_stop_epochs:
            # too many consecutive epochs without surpassing the best model
            print('stopping early')
            break

    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # test_batch = compute_log(eeg.X_test_list)
    test_batch = eeg.X_test_list
    print("Accuracy:", accuracy.eval({X: eeg.X_test_list, Y: eeg.Y_test_list, keep_prob: 1.0}))