import numpy as np



#Step 1: Load data
num_points = 1000
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

import matplotlib.pyplot as plt

##plt.plot(x_data, y_data, 'ro')
##plt.show()


import tensorflow as tf

#Step 2: create placeholders for inputs

#X=tf.placeholder(tf.float32, name='X')
#Y=tf.placeholder(tf.float32, name='Y')


#Step 3: create weight and bais
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

#Step 4: build model to predict y
y = W * x_data + b

#Step 5: use the square error as te loss function
loss = tf.reduce_mean(tf.square(y - y_data))

#Step 6: using gradient descent to mnimize loss
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


#Step 7: call session and initialize w and b
init = tf.initialize_all_variables()
sess= tf.Session()
sess.run(init)


for step in range(8):

    #Step 8: train the model
    sess.run(train)
    w_, b_, loss_= sess.run([W,b,loss])

    #Print the results
    print('step: {:01d} | W : {:.4f} | b : {:.4f} | Loss: {:.4f}'
          .format(step,float(w_),float(b_),float(loss_)))

    #Plolt the results
    fig = plt.figure() 
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, w_ * x_data + b_)
    plt.xlabel('x')
    plt.xlim(-2, 2)
    plt.ylim(0.1, 0.6)
    plt.ylabel('y')
    plt.legend()
    plt.show()

    #Save the plot results
    file_name = str(step)+ '.png'
    fig.savefig( file_name )



