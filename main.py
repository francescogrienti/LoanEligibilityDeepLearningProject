import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from hyperopt import hp, tpe, Trials, fmin, space_eval, STATUS_OK
import numpy as np 
import functions as fun

train_dataset = pd.read_csv('loan-train.xls')
train_dataset.info()

"""
PRE-PROCESSING DATASET
"""
train_dataset.drop(['Gender', 'Education', 'Married', 'Loan_ID'], axis=1, inplace=True)
train_dataset.info()
train_dataset.dropna(inplace=True)
train_dataset.info()
train_dataset['Self_Employed'] = train_dataset['Self_Employed'].map({'Yes': 1.0, 'No': 0.})
train_dataset['Property_Area'] = train_dataset['Property_Area'].map({'Rural': 0., 'Semiurban': 1., 'Urban': 2.})
train_dataset['Dependents'] = train_dataset['Dependents'].map({'0': 0., '1': 1., '2': 2., '3+': 4.})
train_dataset['Loan_Status'] = train_dataset['Loan_Status'].map({'Y': 1.0, 'N': 0.})

for i in train_dataset.columns[1:]:
    train_dataset[i] = (train_dataset[i] - train_dataset[i].min()) / (train_dataset[i].max() - train_dataset[i].min())

train_set = train_dataset.sample(frac=0.9, random_state=1)
test_set = train_dataset.drop(train_set.index)

x_train = train_set[['Dependents', 'Self_Employed', 'ApplicantIncome',
                    'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                    'Credit_History', 'Property_Area']]
y_train = train_set['Loan_Status']

x_test = test_set[['Dependents', 'Self_Employed', 'ApplicantIncome',
                         'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                         'Credit_History', 'Property_Area']]
y_test = test_set['Loan_Status']

#TODO fix images dimensions 

"""
DATA EXPLORING 
"""
i = 0
plt.suptitle('Histograms features', fontsize=24)
for col in train_dataset.columns: 
    plt.subplot(2, 5, i+1)
    plt.hist(train_dataset[col], bins=20, label=col, color='green')
    plt.ylabel('Frequency')
    plt.legend()
    i+=1
plt.rcParams['figure.figsize'] = (30,5)
plt.savefig('./data_ex/histograms.png')
plt.show()

correlation_matrix = train_dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.rcParams['figure.figsize'] = (10,10)
plt.savefig('./data_ex/correlation.png')
plt.show()


def hyperfunc(params):
    model = fun.train_hyper_param_model(x_train, y_train, params, epochs=15)
    test_loss, test_acc = model.evaluate(x_test, y_test)

    return {'loss': test_loss, 'accuracy': test_acc, 'status': STATUS_OK}


def main():

    search_space = {
        'layer1_size': hp.choice('layer1_size', np.arange(1, 30, 1)),
        'layer2_size': hp.choice('layer2_size', np.arange(1, 15, 1)),
        'learning_rate': hp.loguniform('learning_rate', -10, 0)
    }
    trials = Trials()
    best = fmin(hyperfunc, search_space, algo=tpe.suggest, max_evals=5, trials=trials)
    print(space_eval(search_space, best))

    fun.hyper_plots(trials, 'accuracy')
    fun.hyper_plots(trials, 'loss')

    model = fun.loan_elig_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=60, validation_split=0.8, callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    fun.plot_metrics(history, 'loss', 'LOSS')
    fun.plot_metrics(history, 'accuracy', 'ACCURACY')
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    main()









