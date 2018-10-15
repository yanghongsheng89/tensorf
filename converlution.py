# converlution.py
import input_data
import tensorflow as tf
import model
import os

data = input_data.read_data_sets("minist_data",one_hot=True)
with tf.variable_scope("converlution"):
    x = tf.placeholder(tf.float32,[None,784],name='x')
    keep_prob = tf.placeholder(tf.float32)
    y,variables = model.converlution(x,keep_prob)

y_ = tf.placeholder('float',[None,10],name='y')
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver(variables)

with tf.Session() as sess:
    merger_summary = tf.summary.merge_all()
    writer_summary = tf.summary.FileWriter("/tmp/minist_log/1",sess.graph)
    writer_summary.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = data.train.next_batch(50)
        if i%1000 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy %g"%accuracy.eval(feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0}))
    path = saver.save(sess,os.path.join(os.path.dirname(__file__),'data','converlution'),write_meta_graph=False,write_state=False)
    print("save path:",path)
