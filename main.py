import inline as inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
import statsmodels.api as sm
import csv

# import the seaborn module
import seaborn as sns

# reads the data from the csv into a variable
input_file = "Health Insurance Dataset.csv"

# Read in the data with `read_csv()`
insurance_data = pd.read_csv(input_file)

# after analyzing the data for the southwest region, the data from the
# best-fit-line shows that all values corresponding to that region needed
# to be reduced by a factor of 1214.100 for the cost of living in the region

# initialize the iterator variable
t = 0
for i in insurance_data['region']:

    if i == 'southwest':
        insurance_data.at[t, 'charges'] = insurance_data.iloc[t]['charges'] - 1214.100
    t = t + 1

# after analyzing the data for the southeast region, the data from the
# best-fit-line shows that all values corresponding to that region needed
# to be reduced by a factor of 1234.900 for the cost of living in the region

# initialize the iterator variable
t = 0
for i in insurance_data['region']:

    if i == 'southeast':
        insurance_data.at[t, 'charges'] = insurance_data.iloc[t]['charges'] - 1234.900
    t = t + 1

# after analyzing the data for the northeast region, the data from the
# best-fit-line shows that all values corresponding to that region needed
# to be reduced by a factor of 1556.100 for the cost of living in the region

# initialize the iterator variable
t = 0
for i in insurance_data['region']:

    if i == 'northeast':
        insurance_data.at[t, 'charges'] = insurance_data.iloc[t]['charges'] - 1556.600
    t = t + 1

# Transform the data in teh 'sex' Column to a classification, by making male=1 and females=0
t = 0
for i in insurance_data['sex']:

    if i == 'male':
        insurance_data.at[t, 'sex'] = 1
    else:
        insurance_data.at[t, 'sex'] = 0
    t = t + 1

# transform data in the 'smoker' column to a classification, by making yes=1 and no=1
t = 0
for i in insurance_data['smoker']:

    if i == 'yes':
        insurance_data.at[t, 'sex'] = 1
    else:
        insurance_data.at[t, 'sex'] = 0
    t = t + 1

# allocate values in the insurance charges column to the variable
insurance_charges = insurance_data.iloc[0:, 6]

# represents the new variable Y that contains the normalized values
insurance_charges_norm = (insurance_charges - insurance_charges.min()) / \
                         (insurance_charges.max() - insurance_charges.min())

# HW3-Q1. normalized Y(insurance charges) values
insurance_data["insurance_charges_normalized"] = (insurance_charges - insurance_charges.min()) / \
                                                 (insurance_charges.max() - insurance_charges.min())

# ##HW3-Q2.
insurance_data_copy = insurance_data.copy()

#  The training dataset is 75% of the original dataset and it randomly generated
#  The testing dataset is 25% if the original dataset
train_set = insurance_data_copy.sample(frac=0.75, random_state=0)
test_set = insurance_data_copy.drop(train_set.index)

print('\nTraining Data Set')
print(train_set)
print('\nTest Data set')
print(test_set)
# ##HW3-Q2.


# HW3-Q3. plot of the Y-normalized to the dependency children
insurance_data.plot(kind='scatter', x='children', y='insurance_charges_normalized', color='red')
plt.show()

# there is a weak relation between the two plotted variables
# HW3-Q3.


X = train_set[['age', 'bmi']]
Y = train_set['insurance_charges_normalized']
T = test_set[['age', 'bmi']]

# Generate the Regression
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


model = sm.OLS(Y, X).fit()
predictions = model.predict(T)

print_model = model.summary()
print(print_model)

print("The prediction model is displayed ")
print(predictions)
