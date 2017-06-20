# SVM using different kernels
#----------------------------------
#
# Linear Kernel:
# K(x1, x2) = t(x1) * x2
#
# Gaussian Kernel (RBF):
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Generate non-linear data
# make circles function generates data in two concentric circles, samples
# in each circle are labeled with a different class.
(x_vals, y_vals) = datasets.make_circles(n_samples=350, factor=.5, noise=.1)
y_vals = np.array([1 if y == 1 else -1 for y in y_vals])

class1_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class1_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == 1]
class2_x = [x[0] for i, x in enumerate(x_vals) if y_vals[i] == -1]
class2_y = [x[1] for i, x in enumerate(x_vals) if y_vals[i] == -1]

class1 = np.c_[class1_x, class1_y]
class2 = np.c_[class2_x, class2_y]

# Declare batch size
batch_size = 350

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# Create variables for svm
alpha = tf.Variable(tf.random_normal(shape=[1, batch_size]))
b = tf.Variable(tf.random_normal(shape=[1]))

# Apply kernel
#########################################################################################################
# DEFINITION OF PREDICTION KERNEL. COMMENT/UNCOMMENT pred_kernel variable to apply the different
# kernels

# Linear Kernel
kernel = tf.matmul(x_data, tf.transpose(x_data))

# Gaussian (RBF) kernel
# gamma = tf.constant(-50.0)
# dist = tf.reduce_sum(tf.square(x_data), 1)
# dist = tf.reshape(dist, [-1, 1])
# sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
# kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Compute SVM Model
alphaMatrix = tf.matmul(tf.transpose(alpha), alpha)
targetMatrix = tf.matmul(y_target, tf.transpose(y_target))
loss = -1 * (tf.reduce_sum(alpha) - 1/2 * tf.reduce_sum(tf.multiply(kernel, tf.multiply(alphaMatrix, targetMatrix))))

# Create Prediction/classification Kernel

#########################################################################################################
# DEFINITION OF PREDICTION KERNEL. COMMENT/UNCOMMENT pred_kernel variable to apply the different
# kernels

# Linear prediction kernel
pred_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

#Gaussian (RBF) prediction kernel
# rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1,1])
# rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1,1])
# pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
#                       tf.transpose(rB))
# pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
#########################################################################################################

# Prediction and accuracy formulas
A = tf.matmul(tf.multiply(tf.transpose(y_target), alpha), pred_kernel)
b = tf.reduce_mean(y_target - tf.matmul(tf.multiply(tf.transpose(y_target), alpha), kernel))
prediction = tf.sign(A + b)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
batch_accuracy = []

for i in range(2000):
    rand_index = np.random.choice(len(x_vals), size=batch_size, replace=False)
    rand_x = x_vals[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    acc_temp, predictions = sess.run([accuracy, prediction], feed_dict={x_data: rand_x, y_target: rand_y,
                                                                        prediction_grid: rand_x})
    batch_accuracy.append(acc_temp)
    
    if (i+1) % 500 == 0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))
        print('Accuracy = ' + str(acc_temp))

# Create a mesh to plot points and grid predictions
# The grid points act as test points
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]

[grid_predictions] = sess.run(prediction, feed_dict={x_data: rand_x, y_target: rand_y, prediction_grid: grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Class 1')
plt.plot(class2_x, class2_y, 'kx', label='Class -1')
plt.title('SVM Results')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
