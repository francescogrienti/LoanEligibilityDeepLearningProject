import tensorflow as tf 
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK, Trials


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
    plt.savefig('./metrics/' + metric +'.png')
    plt.show()


def loan_elig_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(8)))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(5, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model


def train_hyper_param_model(x_training, y_training, params, epochs):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(8)))
    model.add(tf.keras.layers.Dense(params['layer1_size'], activation='relu'))
    model.add(tf.keras.layers.Dense(params['layer2_size'], activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    model.fit(x_training, y_training, epochs=epochs)

    return model


def hyper_plots(trials, metrics:str):
    
    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    xs = [t['tid'] for t in trials.trials]
    ys = [t['result'][metrics] for t in trials.trials]
    ax1.set_xlim(xs[0]-1, xs[-1]+1)
    ax1.scatter(xs, ys, s=20)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(metrics)

    xs = [t['misc']['vals']['layer1_size'] for t in trials.trials]
    ys = [t['result'][metrics] for t in trials.trials]

    ax2.scatter(xs, ys, s=20)
    ax2.set_xlabel('Layer1_size')
    ax2.set_ylabel(metrics)

    xs = [t['misc']['vals']['layer2_size'] for t in trials.trials]
    ys = [t['result'][metrics] for t in trials.trials]

    ax3.scatter(xs, ys, s=20)
    ax3.set_xlabel('Layer2_size')
    ax3.set_ylabel(metrics)

    xs = [t['misc']['vals']['learning_rate'] for t in trials.trials]
    ys = [t['result'][metrics] for t in trials.trials]

    ax4.scatter(xs, ys, s=20)
    ax4.set_xlabel('learning_rate')
    ax4.set_ylabel(metrics)
    plt.savefig('./hyper_opt/' + metrics +'.png')
    plt.show()