import tensorflow as tf

#x_train = [1, 2, 3]
#y_train = [1, 2, 3]
x = tf.placeholder(tf.float32, shape=[None], name = "x")
y = tf.placeholder(tf.float32, shape=[None], name = "y")


W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

hypothesis = x * W + b
cost = tf.reduce_mean(tf.square(hypothesis - y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, W_val, b_val,_ = sess.run([cost, W, b, train], feed_dict={x:[1,2,3], y:[1,2,3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

print(sess.run(hypothesis, feed_dict={x:[5]}))
print(sess.run(hypothesis, feed_dict={x:[2.5]}))
print(sess.run(hypothesis, feed_dict={x:[1.5, 5.5]}))
