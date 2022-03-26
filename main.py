import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.regression import r2_score
import matplotlib.pyplot as plt

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

plt.figure()
plt.scatter(X_train, y_train, label='training data')
plt.scatter(X_test, y_test, label='test data')
plt.legend(loc=4);
plt.show()

r2_train = np.zeros(10)
r2_test = np.zeros(10)

for i in range(10):
    poly = PolynomialFeatures(degree=i)
    # Train and score x_train
    X_poly = poly.fit_transform(X_train.reshape(11,1))
    linreg = LinearRegression().fit(X_poly, y_train)
    r2_train[i] = linreg.score(X_poly, y_train);

    X_test_poly = poly.fit_transform(X_test.reshape(4,1))
    r2_test[i] = linreg.score(X_test_poly, y_test)
    print(r2_train, r2_test)

r2_train.plot()
plt.show()

r2_scores = answer_two()
training_scores = r2_scores[0]
testing_scores = r2_scores[1]
difference = training_scores - testing_scores
Good_Generalization = np.sort(difference)[0]
Overfitting = np.sort(difference)[difference.shape[0]-1]
Underfitting = np.sort(testing_scores)[0]
print(Good_Generalization)
print(Overfitting)
print(Underfitting)


poly = PolynomialFeatures(degree=12)
X_poly = poly.fit_transform(X_train.reshape(11,1))
linreg = LinearRegression().fit(X_poly, y_train)
X_test_poly = poly.fit_transform(X_test.reshape(4,1))
LinearRegression_R2_test_score = linreg.score(X_test_poly, y_test)
lasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_poly, y_train)
Lasso_R2_test_score = lasso.score(X_test_poly, y_test)