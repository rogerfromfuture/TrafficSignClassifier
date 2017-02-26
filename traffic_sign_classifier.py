import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import tensorflow as tf

# Global configuration
plt.interactive(True)

# Load pickled data
training_file = './data/training/train.p'
validation_file = './data/training/valid.p'
testing_file = './data/training/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results
# Number of training examples
n_train = np.shape(X_train)[0]

# Number of testing examples
n_test = np.shape(X_test)[0]

# Shape of an traffic sign image
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset
unique_items, unique_items_counts = np.unique(y_train, return_counts=True)
n_classes = unique_items.size

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.
sign_dict = dict(zip(unique_items, unique_items_counts))
x = np.fromiter(sign_dict.keys(), dtype=float)
y = np.fromiter(sign_dict.values(), dtype=float)
plt.plot(x, y, ".r")

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()
print(image.shape)
plt.figure(figsize=(1, 1))
plt.imshow(image)
print(y_train[index])

plt.show()

#Step 2: Design and Test a Model Architecture
### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

### Define your architecture here.
### Feel free to use as many code cells as needed.
### Define your architecture here.
EPOCHS = 30
BATCH_SIZE = 64

from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 4: Fully Connected. Input = 84. Output = 60.
    fc22_W = tf.Variable(tf.truncated_normal(shape=(84, 60), mean=mu, stddev=sigma))
    fc22_b = tf.Variable(tf.zeros(60))
    fc22 = tf.matmul(fc2, fc22_W) + fc22_b

    # SOLUTION: Activation.
    fc22 = tf.nn.relu(fc22)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(60, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc22, fc3_W) + fc3_b

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './trafficsign')
    print("Model saved")

#Step 3: Test a Model on New Images
### Load the images and plot them here.
num_of_images = 5
start_index = random.randint(0, len(X_test) - num_of_images)
end_index = start_index + num_of_images
X_test_five = X_test[start_index:end_index]
y_test_five = y_test[start_index:end_index]

for i in range(0, num_of_images):
    image = X_test_five[i].squeeze()
    plt.figure(figsize=(1, 1))
    plt.imshow(image)
    print(image.shape)
    print(y_test_five[i])

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
softmax = tf.nn.softmax(logits)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    result = sess.run(softmax, feed_dict={x: X_test_five})
    values, indices = tf.nn.top_k(result, num_of_images)
    predictions = sess.run(indices)
    print("Top {} predictions are {}".format(num_of_images, predictions))
    print("The real values are {}".format(y_test_five))

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
### Calculate the accuracy for these 5 new images.
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    test_accuracy = evaluate(X_test_five, y_test_five)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    result = sess.run(softmax, feed_dict={x: X_test_five})
    values, indices = tf.nn.top_k(result, num_of_images)
    probabilities = sess.run(values)
    print("Top {} probabilities are {}".format(num_of_images, probabilities))