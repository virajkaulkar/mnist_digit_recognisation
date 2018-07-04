# -*- coding: utf-8 -*-

#import data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#set tuning params
learning_rate = 0.01
training_iterations = 30
batch_size = 100
display_step = 2

#TF graph input

x = tf.placeholder("float", [None, 784]) #data image of shape 28*28 = 784
y = tf.placeholder("float", [None, 10]) #0-9 digits recognization 10 classes

#Create Model
#set Model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
    #construct linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)

#Add summary ops to collect data
w_h = tf.summary.histogram("Weights", W)
b_h = tf.summary.histogram("Biases", b)


with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(x, 300, name="hidden1",
                              activation=tf.nn.relu)
    #hidden2 = tf.layers.dense(hidden1, 300, name="hidden2",
     #                         activation=tf.nn.relu)
    logits = tf.layers.dense(hidden1, 10, name="outputs")

#More name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
    #minimise error using cross entropy
    #cross entropy
    #cost_function = -tf.reduce_sum(y*tf.log(model))
    xentropy =tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    cost_function = tf.reduce_mean(xentropy, name='loss')
    
    #create summary to monitor cost function
    tf.summary.scalar("cost function", cost_function)
    
with tf.name_scope("train") as scope:
    #gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
        
#Initialize all variables
init = tf.initialize_all_variables()

#mearge all summaries into a single operator
merged_summary_op = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)

#Launch a graph
with tf.Session() as sess:
    sess.run(init)
    #set the logs writer to the folder /tmp/tf_logs
    summary_writer = tf.summary.FileWriter("tensorflow_mnist/logs", sess.graph)
    
    #training cycles
    for iteration in range(training_iterations):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        #loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #fit training using batch data
            sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
            #compute the avarage loss
            avg_cost += sess.run(cost_function, feed_dict={x:batch_xs, y:batch_ys})/total_batch
            #write cost for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x:batch_xs, y:batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + 1)
        #Display logs per iteration step
        if iteration%display_step == 0:
            print("Iteration: ", '%04d' %(iteration+1), "cost = ", "{:.9f}".format(avg_cost))
        print("Tuning completed")
    #test the model
    predictions = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
    #calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy: ", accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))             
                


