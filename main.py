import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import functions as fun
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


dataset = pd.read_csv('loan-train.xls')
dataset.info()

"""
PRE-PROCESSING DATASET
"""

dataset.drop(['Gender', 'Education', 'Married', 'Loan_ID'], axis=1, inplace=True)
dataset.info()
dataset.dropna(inplace=True)
dataset.info()
dataset['Self_Employed'] = dataset['Self_Employed'].map({'Yes': 1.0, 'No': 0.})
dataset['Property_Area'] = dataset['Property_Area'].map({'Rural': 0., 'Semiurban': 1., 'Urban': 2.})
dataset['Dependents'] = dataset['Dependents'].map({'0': 0., '1': 1., '2': 2., '3+': 4.})
dataset['Loan_Status'] = dataset['Loan_Status'].map({'Y': 1.0, 'N': 0.})

for i in dataset.columns[1:]:
    dataset[i] = (dataset[i] - dataset[i].min()) / (dataset[i].max() - dataset[i].min())

train_set = dataset.sample(frac=0.8, random_state=1)
test_set = dataset.drop(train_set.index)
valid_set = train_set.sample(frac=0.2, random_state=1)

x_train = train_set[['Dependents', 'Self_Employed', 'ApplicantIncome',
                    'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                    'Credit_History', 'Property_Area']]
y_train = train_set['Loan_Status']
x_valid = valid_set[['Dependents', 'Self_Employed', 'ApplicantIncome',
                         'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                         'Credit_History', 'Property_Area']]
y_valid = valid_set['Loan_Status']
x_test = test_set[['Dependents', 'Self_Employed', 'ApplicantIncome',
                         'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                         'Credit_History', 'Property_Area']]
y_test = test_set['Loan_Status']

"""
DATA EXPLORATION
"""

i = 0
plt.figure(figsize=(20, 7))
plt.suptitle('Histograms features', fontsize=24)
for col in dataset.columns: 
    plt.subplot(2, 5, i+1)
    plt.hist(dataset[col], bins=20, label=col, color='green')
    plt.ylabel('Frequency')
    plt.legend()
    i+=1
plt.savefig('./data_ex/histograms.png')
plt.show()

plt.figure(figsize=(20, 10))
correlation_matrix = dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('./data_ex/correlation.png')
plt.show()

"""
MAIN 
"""

def main():

    tf.random.set_seed(1)
    print('--- Initialising neural net model ---')
    model = fun.loan_elig_model()
    print('--- Training with starting hyperparameters set ---')
    history = model.fit(x_train, y_train, epochs=60, validation_data=(x_valid, y_valid), batch_size=32)
    fun.plot_metrics(history, 'loss', 'Loss')
    fun.plot_metrics(history, 'accuracy', 'Accuracy')
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy on test set with no hyperparameters tuning {:.2f}".format(accuracy))
    print("Loss on test set with no hyperparameters tuning {:.2f}".format(loss))
    print('--- Hyperparameters tuning using RandomSearch ---')
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=fun.loan_elig_model, verbose=0)
    layer1_size = [2,10,20,30,40,50,60,70,80,90,100]
    layer2_size = [3,5,10,15,20,25,30,35,40,45,50,55,60] 
    learning_rate = [0.01, 0.001, 0.0001]
    batch_size = [4,8,16,32]
    epochs = [10,20,30,40,50,60,70,80,100]
    grid = dict(
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )   
    searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=5, param_distributions=grid, n_iter=50, scoring=('accuracy'))
    search_results = searcher.fit(x_valid, y_valid)
    best_score = search_results.best_score_
    best_params = search_results.best_params_
    print('Best score is {:.2f} using {}'.format(best_score, best_params))
    best_model = search_results.best_estimator_
    accuracy = best_model.score(x_test, y_test)
    print("Accuracy on test set {:.2f}".format(accuracy))
    
    # Logistic regression
    model2 = LogisticRegression()
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('----- Logistic regression -----')
    print('Accuracy:', accuracy)
    # Decision tree classifier
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print('----- Decision tree classifier -----')
    print('Accuracy:', acc)


if __name__ == "__main__":
    main()