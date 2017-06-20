from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class color:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 500
display_step = 1


# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define the model
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=1))

# Accuracy
rightPredictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(rightPredictions, tf.float32))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Training cycle
cost_list = []
acc_list = []
for epoch in range(training_epochs):
    avg_cost = 0.
    avg_acc = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
        _, c2 = sess.run([optimizer, acc], feed_dict={x: batch_xs, y: batch_ys})
        # Compute average loss
        avg_cost += c / total_batch      
        # Compute average acc 
        avg_acc += c2 / total_batch      
    # Display logs per epoch step (cost)
    if (epoch + 1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        cost_list.append(avg_cost)
    # Display lpgs per epoch step (acc)
    if (epoch + 1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "acc=", "{:.9f}".format(avg_acc))
        acc_list.append(avg_acc)
        
        
print("Optimization Finished!")

# EXERCISE L.R.2: PLOT COST AND ACCURACY PER EPOCH
plt.plot(range(1,training_epochs+1), cost_list)
plt.title('Cost reduction per optimisation epoch')
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()

plt.plot(range(1,training_epochs+1), acc_list)
plt.title('Accuracy increase per optimisation epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.show()

plt.plot(cost_list, acc_list)
plt.title('Cost vs. Accuracy (per epoch)')
plt.ylabel('Accuracy')
plt.xlabel('Cost')
plt.show()


# EXERCISE L.R.3: COMPUTE THE ACCURACY FOR THE TEST SET
# Note that in mnist set: test data and labels are mnist.test.images and mnist.test.labels
## Evaluating the model
rightPredictions = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(rightPredictions, tf.float32))
test_acc = sess.run(acc, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("test accuracy is: ", test_acc)

# show total number of correct/incorrect 
n = len(mnist.test.images)
num_correct = (test_acc * n)
num_incorrect = n - (test_acc * n)
print("number of" + color.BOLD + " incorrect" + color.END + " test classifications is: ", int(round(num_incorrect)), "out of ", n)

# actual/predicted 3 samples 

