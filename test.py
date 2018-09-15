from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from sklearn.model_selection import train_test_split
x_train1, x_valid, y_train1, y_valid = train_test_split(x_train, y_train, test_size=0.175)
