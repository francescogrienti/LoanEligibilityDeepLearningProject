import tensorflow as tf 
import matplotlib.pyplot as plt


def plot_metrics(history, metric:str, title:str):
    plt.figure(figsize=(20, 10))
    x = range(1, len(history.history[metric])+1)
    yt = history.history[metric]
    yv = history.history['val_'+ metric]
    plt.plot(x, yt, label='Training ' + metric)
    plt.plot(x, yv, label='Validation ' + metric)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.savefig('/home/francescogrienti/DL/DeepLearning/LoanEligibilityDeepLearningProject/metrics/' + metric +'.png')
    plt.show()


def loan_elig_model(layer1_size=2, layer2_size=3, learning_rate=0.001):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(8)))
    model.add(tf.keras.layers.Dense(layer1_size, activation='relu'))
    model.add(tf.keras.layers.Dense(layer2_size, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

    return model
