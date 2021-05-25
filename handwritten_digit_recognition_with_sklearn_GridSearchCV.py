# Building a supervised machine learning model to recognize handwritten digits (e.g., 0 to 9) using the sklearn library

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Reading test.csv and train.csv
train = pd.read_csv('train.csv')
print(train.head())
print(train.info())
test = pd.read_csv('test.csv')
print(test.head())
print(test.info())
# note: train.csv and test.csv have been loaded as DataFrames train and test, respectively. train has 42,000 observations; test has 28,000 observations
# note: each row represents a flattened array of a 28x28 image (i.e. 784 columns total), and each column represents a pixel value (between 0 [white] and 255 [black])

# Examine the 100th observation from train and double-check against its corresponding label
hundredth_pic = train.iloc[99, 1:]
hundredth_pic_array = hundredth_pic.values
hundredth_pic_array = hundredth_pic_array.reshape(28,28)
plt.imshow(hundredth_pic_array)
plt.show()
hundredth_pic_label = train.iloc[99, 0]
print(hundredth_pic_label)
# note: the 100th observation label is '5' and the corresponding reshaped image shows '5'

# Splitting the train data into train samples (75%) & test samples (25%)
train_x = train.iloc[:,1:] # isolates all 784 pixel columns and skips the 0th column (labels)
train_y = train.iloc[:,0] # isolates the 0th column (labels)
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.25, stratify=train_y, random_state=42)

# Instantiate the Random Forest
rf = RandomForestClassifier(random_state=42)

# Using GridSearchCV to evaluate model accuracy
params_rf = {'n_estimators': [200, 300]}
grid_rf = GridSearchCV(estimator=rf, param_grid=params_rf, cv=3, scoring='accuracy')
grid_rf.fit(X_train, y_train)
best_hyperparams = grid_rf.best_params_
print(best_hyperparams)
best_model = grid_rf.best_estimator_

# Predict y_pred labels using X_test and calculate accuracy, confusion matrix, and classification report
y_pred = best_model.predict(X_test)
best_model.score(X_train, y_train) # training sample accuracy
best_model.score(X_test, y_test) # training sample accuracy

confusion_matrix = confusion_matrix(y_test, y_pred) # rows represent the real classes, columns represent the predicted classes
sns.heatmap(confusion_matrix, annot=True, annot_kws={"fontsize":8})
plt.show()

print(classification_report(y_test, y_pred))
# note: the model has an accuracy of 96%

# Cross-check 10 values from the testing sample of train with its predicted labels
print(y_test[0:10])
print(y_pred[0:10])
print(y_test[0:10] == y_pred[0:10])

# Predict the labels for the test DataFrame and check a couple images & their predicted labels
test_pred = best_model.predict(test)

first_ten_predicted_labels = []
for x in range(10):
    image = test.iloc[x,:].values
    image = image.reshape(28,28)
    plt.imshow(image)
    plt.show()
    first_ten_predicted_labels.append(test_pred[x])
print(first_ten_predicted_labels) # compare this list to the shown images from the above for loop