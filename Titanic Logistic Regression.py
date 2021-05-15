import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv('passengers.csv')
print(passengers)
# Update sex column to numerical
# change the words male = 0, female = 1
passengers['Sex'] = passengers['Sex'].map({'male': 0, 'female': 1})

# Fill the nan values in the age column
print(passengers['Age'].values)
# got nan(null values), need to fill them up with mean value
passengers['Age'].fillna(value=round(np.mean(passengers['Age'])), inplace=True)
print(passengers['Age'].values)

# Create a first class column, add new column called first class that store 1 if the Pclass is 1 else store 0
passengers['FirstClass'] = passengers['Pclass'].apply(
    lambda x: 1 if x == 1 else 0)
print(passengers)
# Create a second class column, add new column called second class that store 1 if Pclass is 2 else store 0
passengers['SecondClass'] = passengers['Pclass'].apply(
    lambda x: 1 if x == 2 else 0)
print(passengers)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
Survival = passengers[['Survived']]
# Perform train, test, split
train_features, test_features, train_labels, test_labels = train_test_split(
    features, Survival, test_size=0.2, train_size=0.8)
# normalize the data
# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.fit_transform(test_features)

# Create and train the model
model = LogisticRegression()
model.fit(train_features, train_labels)

# Score the model on the train data
train_score = model.score(train_features, train_labels)
print(train_score)

# Score the model on the test data
test_score = model.score(test_features, test_labels)
print(test_score)

# Analyze the coefficients
print(model.coef_)
# coef_ answer will be for ['Sex', 'Age', 'FirstClass', 'SecondClass']
# sex plays most important part , second is FirstClass
# Sample passenger features
Jack = np.array([0.0, 20.0, 0.0, 0.0])
Rose = np.array([1.0, 17.0, 1.0, 0.0])
You = np.array([0.0, 21.0, 0.0, 1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])
# normalised the sample datasets
# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)
# In layman's terms, fit_transform means to do some calculation and then do transformation (say calculating the means of columns from some data and then replacing the missing values). So for training set, you need to both calculate and do transformation.

# But for testing set, Machine learning applies prediction based on what was learned during the training set and so it doesn't need to calculate, it just performs the transformation.

# Make survival predictions!
predicted_result = model.predict(sample_passengers)
print(predicted_result)
predicted_result_probability = model.predict_proba(sample_passengers)
# get the probability for u die or survived
print(predicted_result_probability)
