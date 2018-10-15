# ala.py
import input_data
import tensorflow as tf
import model
import os

data = input_data.read_data_sets("minist_data",one_hot=True)

with tf.variable_scope('regression'):
    x = tf.placeholder(tf.float32,[None,784])
    y,variables = model.regression(x)

y_ = tf.placeholder('float',[None,10])

cross_entry = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entry)

credict_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accurary = tf.reduce_mean(tf.cast(credict_prediction,tf.float32))
saver =tf.train.Saver(variables)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(10000):
        batch_xs,batch_ys = data.train.next_batch(100)
        session.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
    print(session.run(accurary,feed_dict={x:data.test.images,y_:data.test.labels}))
    path = saver.save(session,os.path.join(os.path.dirname(__file__),'data','regression'),write_meta_graph=False,write_state=False,)

    print("save %s",path)

