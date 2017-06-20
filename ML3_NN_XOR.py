'''
A Multilayer Neural Network implementation using TensorFlow library.

This example is using XOR data
'''

import tensorflow as tf

# XOR definition
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y = [[0], [1], [1], [0]]

# Neural Network Parameters
N_STEPS = 20000
N_EPOCH = 1000
N_TRAINING = len(X)

N_INPUT_NODES = 2
N_OUTPUT_NODES = 1
LEARNING_RATE = 0.2


x = list(range(1,6,1))
final_inf = []

for N_HIDDEN_NODES in x:
# Create placeholders for variables and define Neural Network structure
    x_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_INPUT_NODES], name="x-input")
    y_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_OUTPUT_NODES], name="y-input")
    
    
    theta1 = tf.Variable(tf.random_uniform([N_INPUT_NODES, N_HIDDEN_NODES], -1, 1), name="theta1")
    theta2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES, N_OUTPUT_NODES], -1, 1), name="theta2")
    
    bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")
    bias2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias2")
    
    # Use a sigmoidal activation function
    layer1 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
    output = tf.sigmoid(tf.matmul(layer1, theta2) + bias2)
    
    # Cross Entropy cost function
    cost = - tf.reduce_mean((y_ * tf.log(output)) + (1 - y_) * tf.log(1.0 - output))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(N_STEPS):
        sess.run(train_step, feed_dict={x_: X, y_: Y})
        if i % N_EPOCH == 0:
            print('Batch ', i)
            print('Inference ', sess.run(output, feed_dict={x_: X, y_: Y}))
            inf = sess.run(output, feed_dict={x_: X, y_: Y})
            print('Cost ', sess.run(cost, feed_dict={x_: X, y_: Y}))
            cost = sess.run(cost, feed_dict={x_: X, y_: Y}
            final_inf.append(N_HIDDEN_NODES)
            final_inf.append(inf)
            final_inf.append(cost)

print(final_inf)
    # run whole code iterating through number of nodes 
    # append final inference to list 