"""
Created on Mon Nov 22 15:23:02 2021

@author: 13125

1. Mall Customers Dataset
The Mall customers dataset contains information about people visiting the mall. 
The dataset has gender, customer id, age, annual income, and spending score. 
It collects insights from the data and group customers based on their behaviors.

"""
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# Load dataset
names = ['customerID', 'Genre', 'Age', 'Annual Income(k$)', 'Spending Score(1-100)']
dataset = read_csv('Mall_Customers.csv', names = names)

# shape
print(dataset.shape)

# head
print(dataset.head(5))

# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('Spending Score(1-100)').size())

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# Split-out validation dataset
# 50% Test 50% Training
# Import LabelEncoder

array = dataset.values
X = array[:,2:5]
y = array[:,1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.50, random_state=1)

print("\nX_train:\n", X_train) 
print("\nY_train:\n", Y_train)   

# Spot Check Algorithms
models = []
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []

# Create a Gaussian classifier
# Make prediction on validation dataset (test data set)
# Train the model using training sets: .fit()

model = GaussianNB()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print('\nPredictions:\n', predictions)

# Evaluate predictions
print("\nAccuracy score: ",accuracy_score(Y_validation, predictions))
#print("\nConfusion matrix:\n",confusion_matrix(Y_validation, predictions))
print("\nClassification report:\n",classification_report(Y_validation, predictions))

for name, model in models:
    print('\n',name, results)
    kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
    print('\n',type(kfold))
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('\n%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
