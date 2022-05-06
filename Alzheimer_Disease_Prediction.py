#!/usr/bin/env python
# coding: utf-8

# # Predicting Alzheimer disease using machine learning
# 
# this notebook looks into using various Python-based machine learning and data science libraries in an attempt to build machine learning model capable of predicting whether or not someone has Alzheimer disease based on their medical attributes
# 
# we're going to take the following approach
# 
#     1. Problem definition
#     2. Data
#     3. Evaluation
#     4. Features
#     5. Modeling
#     4. Building a Predictive System
#     
# ## 1. Problem Definition
# 
# In a statement
# > Given clinical parameters about a patient, can we predict whether or not they have Alzheimer disease
# 
# ## 2. Data
# 
# the original data come from Alzheimer Features For Analysis on Kaggle
# URL: https://www.kaggle.com/code/hyunseokc/detecting-early-alzheimer-s/notebook?fbclid=IwAR1Uscvx557m0PYmz9C1GXq59nIkCYLQHvtGOR6EaWHT3UaasC3jkJBUJ_k
# 
# ## 3.Evaluation 
# > if we can reach 95% accuracy at prediction whether or not a patient has heart disease during the proof of concept, we'll pursue the project
# 
# ## 4.features
# 
# *Create data dictionary *
# 
# > Group --> Class
# * 72 of the subjects were grouped as 'Nondemented' throughout the study.
# * 64 of the subjects were grouped as 'Demented' at the time of their initial visits and remained so throughout the study.
# * 14 subjects were grouped as 'Nondemented' at the time of their initial visit and were subsequently characterized as 'Demented' at a later visit. These fall under the 'Converted' category.
# 
# > Age --> Age
# 
# 
# > EDUC --> Years of Education
# 
# > SES --> Socioeconomic Status / 1-5
# 
# > MMSE --> Mini Mental State Examination : is a widely used test of cognitive function among the elderly; it includes tests of orientation, attention, memory, ...
# 
# > CDR --> Clinical Dementia Rating
# 
# <img src="Images/CDR.png" />
# 
# > eTIV --> Estimated total intracranial volume
# 
# > nWBV --> Normalize Whole Brain Volume
# 
# > ASF --> Atlas Scaling Factor
# 
# # Preparing the tools
# 
# we're going to use pandas , Matplotlib , Numpy and some of scikitLearn librairy for data analysis and manipulation
# 
#     

# In[1]:


# Import all the tools we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# we want our plots to appear inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# the diffrent models (just to choose the best one)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Models from Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Model Evaluations
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score,  GridSearchCV


# # Load DATA

# In[2]:


# Read data
alzheimer_data = pd.read_csv('Alzheimer.csv')
# Show data 
alzheimer_data


# In[3]:


# print the first 5 rows of DataFrame
alzheimer_data


# In[4]:


# print last 5 rows of the DataFrame

alzheimer_data.tail()


# In[5]:


# Number of rows and columns in the dataset

alzheimer_data.shape


# In[6]:


# getting some info about the data

alzheimer_data.info()


# ### Make sure it's all numeric
# 

# In[7]:


# view types of the data
alzheimer_data.dtypes


# * we need to change it like the following:
# 
# > {"Demented": 1, "Nondemented": 0, "Converted": 2}
# 
# > {"F": 0, "M": 1}

# In[8]:


alzheimer_data['Group'] = alzheimer_data['Group'].replace(['Demented', 'Nondemented', 'Converted'],[1,0, 2])
alzheimer_data['M/F'] = alzheimer_data['M/F'].replace(['F', 'M'],[0,1])


# In[9]:


alzheimer_data


# ### Missing values
# 1. Fill them with some value (also khnow as imputation) .
# 2. Remove the samples with missing data alltogether

# In[10]:


# see data with missing values (NaN)
alzheimer_data


# In[11]:


# show how much missing values there are
alzheimer_data.isna().sum()


# In[12]:


### Fill missing data with Pandas

alzheimer_data["SES"].fillna(alzheimer_data["SES"].mean(), inplace= True)
alzheimer_data["MMSE"].fillna(alzheimer_data["MMSE"].mean(), inplace= True)


# In[13]:


# Check our dataframe again
alzheimer_data.isna().sum()


# In[14]:


len(alzheimer_data)


# In[15]:


# show data
alzheimer_data


# In[16]:


alzheimer_data['Group'].value_counts()


# In[17]:


alzheimer_data['Group'].value_counts().plot(kind='bar', color=["salmon", "lightblue"])


# # Some reading about the features
# 
# ## Alzheimer disease Frequency according to Sex

# In[18]:


# Compare Group column with sex column
pd.crosstab(alzheimer_data.Group, alzheimer_data['M/F'])


# In[19]:


# Create a plot of crosstab
pd.crosstab(alzheimer_data.Group, alzheimer_data['M/F']).plot(kind="bar",
                                   figsize=(10, 6),
                                   color=["salmon", "lightblue"])


plt.title("Alzheimer Disease Frequency for sex")
plt.xlabel("0 = No Disease, 1=Disease")
plt.ylabel("Amount")
plt.legend(["Male", "Female"])
plt.xticks(rotation=0);


# ## Alzheimer disease Frequency according to Age

# In[20]:


alzheimer_data.Age.value_counts()


# In[21]:


# Compare Group column with age column
pd.crosstab(alzheimer_data.Group, alzheimer_data.Age)


# In[22]:


# Create another figure
plt.figure(figsize=(10, 6))

# Scatter with positive examples
plt.scatter(alzheimer_data.Age[alzheimer_data.Group ==1],
           alzheimer_data.MMSE[alzheimer_data.Group==1],
           c="salmon")

# Scatter with negative examples
plt.scatter(alzheimer_data.Age[alzheimer_data.Group ==0],
           alzheimer_data.MMSE[alzheimer_data.Group==0],
           c="lightblue")

# Add some helpful info
plt.title("Alzhiemer Disease in function of Age and Mini Mental State Examination")
plt.xlabel("Age")
plt.ylabel("Mini Mental State Examination")
plt.legend(["Disease", "No Disease"])


# In[23]:


# Make a correlation matrix
alzheimer_data.corr()


# In[24]:


# Let's make our correlation matrix a little prettier
corr_matrix = alzheimer_data.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                annot=True,
                linewidths=0.5,
                fmt=".2f",
                cmap="Blues")

bottom, top = ax.get_ylim()
ax.set_ylim(bottom+ 0.5, top-0.5)


# # Spliting the Features and group

# In[25]:


X = alzheimer_data.drop(columns='Group', axis=1)
Y = alzheimer_data['Group']


# In[26]:


X


# In[27]:


Y


# # Splitting the data into Training data & Test Data

# In[28]:


np.random.seed(42)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state=2)
# test_size: mean how much percentage of the data you want to test


# In[29]:


X_train.shape, X_test.shape


# In[30]:


Y_train.shape, Y_test.shape


# In[31]:


X.shape, Y.shape


# Now we've got our data split into training and test sets, it's time to build a machine learning model
# 
# we'll train if(find the patterns) on the training set
# 
# And we'll test it (use the patterns) on the test set
# 
# We're going to choose one the 3 different machine learning models:
# 
# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier

# In[32]:


# Put models in a dictionnary
models = {"Logistic Regression": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, Y_train, Y_test):
    """
    Fits and evaluates given machine learning models
    models: a dict of different Scikit-Learn machine learning models
    X_train: training data (no labels)
    X_test: testing data (no labels)
    Y_train : training labels
    Y_test : test labels
    """
    # Set random seed
    np.random.seed(42)
    #Make a dictionary to keep models scores
    model_scores = {}
    # loop through Models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, Y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, Y_test)
    return model_scores


# In[33]:


print(fit_and_score(models, X_train, X_test, Y_train, Y_test))
model_compare = pd.DataFrame(fit_and_score(models, X_train, X_test, Y_train, Y_test), index=["accuracy"])
model_compare.plot.bar()


# => from the plot up we can decide that Random Forest is the best model for this dataframe

# # Model training

# > Random Forest Classifier

# In[34]:


model = RandomForestClassifier()


# In[35]:


# Training the RandomForestClassifier model with Training data
model.fit(X_train, Y_train)


# # Model Evaluation

# > Accurancy Score

# In[36]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[37]:


print(f'Accuracy on Training data : {training_data_accuracy}')


# In[38]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[39]:


print(f'Accuracy on Test data : {test_data_accuracy}')


# In[40]:


# Make predictions with tuned model
y_preds = model.predict(X_test)
y_preds


# In[41]:


Y_test


# what is ROC curve?
# An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds.

# In[42]:


# View accuracy score
accuracy_score(Y_test, y_preds)


# => This model has an accuracy score of 92% on the test data. That seems pretty impressive, but accuracy is not a great measure of classifier performance when the classes are imbalanced
# 
# ## Confusion matrix
# 
# A confusion matrix is a way to express how many of a classifier’s predictions were correct, and when incorrect, where the classifier got confused. 

# In[43]:


matrix = confusion_matrix(Y_test, y_preds)
matrix


# In[44]:


sns.set(font_scale = 2.5)

def plot_conf_mat(Y_test, y_preds):
    """
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax= plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(Y_test, y_preds),
                    annot=True,
                    cbar=False)
    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    
plot_conf_mat(Y_test, y_preds)


# Now it’s easy to see that our classifier struggled at predicting
# ## Classification report
# To get even more insight into model performance, we should examine other metrics like precision, recall, and F1 score

# In[45]:


# View the classification report for test data and predictions
print(classification_report(Y_test, y_preds))


# Precision is high, meaning that the model was careful to avoid labeling things. 

# ### Calculate evaluation metrics using cross-validation
# 
# we're going to calculate precision, recall and f1-score of our model using cross-validation and to do so we'll be using 'cross_val_scores()'

# In[46]:


# Cross-validated accuracy
cv_acc = cross_val_score(model,X,Y, cv=5, scoring="accuracy").mean()
cv_acc


# In[47]:


# Cross-validated precision
cv_precision = cross_val_score(model,
                               X,
                               Y, 
                               cv=5, 
                               scoring="precision_micro"
                               ).mean()

cv_precision


# In[48]:


# Cross-validated recall
cv_recall = cross_val_score(model, X, Y, scoring="recall_micro", cv = 5).mean()
cv_recall


# In[49]:


# Cross-validated f1-score
cv_f1 = cross_val_score(model,
                       X,
                       Y,
                       cv=5,
                       scoring="f1_micro").mean()
cv_f1


# In[50]:


# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({
    "Accuracy":cv_acc,
    "Precision":cv_precision,
    "Recall":cv_recall,
    "F1": cv_f1
}, index=[0])
cv_metrics.T.plot.bar(title="Cross-validated classification metrics",
                     legend=False)


# ## Feature importance
# 
# Feature importance is another as asking, " which featurs contributed most to the outcomes of the model and how did they contribute ?"
# 
# Finding feature importance is different for each machine learning model. One way to find feature importance is to search for "(MODEL NAME) feature importance".
# 
# Let's find the feature importance for our RandomForestClassifier model...

# In[51]:


# Match coef's of feature to columns
feature_dict = dict(zip(X.columns, list(model.feature_importances_)))
feature_dict


# In[52]:


# Visualize feature importance
feature_data = pd.DataFrame(feature_dict, index=[0])
feature_data.T.plot.bar(title="Feature Importance", legend=False)


# In[ ]:





# # Example: Building a Predictive System

# In[54]:


input_data = (0,69,12,10,29,0,1365,0.783,1.286)

# change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicitng for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if (prediction[0] == 0):
    print('The Person does not have a Alzheimer disease')
elif prediction[0] == 1:
    print('This Person has Alzheimer disease')
else:
    print('This Person start to have Alzheimer')


# In[90]:


from flask import Flask,request,jsonify
import pickle

app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    #(Gender,Age,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF)
    Gender = int(request.form.get('Gender'))
    print(Gender)
    Age = int(request.form.get('Age'))
    EDUC = float(request.form.get('EDUC'))
    SES = float(request.form.get('SES'))
    MMSE = float(request.form.get('MMSE'))
    CDR = float(request.form.get('CDR'))
    eTIV = float(request.form.get('eTIV'))
    nWBV = float(request.form.get('nWBV'))
    ASF = float(request.form.get('ASF'))
    input_query = tuple((Gender,Age,EDUC,SES,MMSE,CDR,eTIV,nWBV,ASF))
    print(input_query)
    # change the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_query)

    # reshape the numpy array as we are predicitng for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if (prediction[0] == 0):
        result = 'The Person does not have a Alzheimer disease'
    elif prediction[0] == 1:
        result = 'This Person has Alzheimer disease'
    else:
        result = 'This Person start to have Alzheimer'
    return jsonify({'Result':str(result)})


# In[100]:


if __name__ == '__main__':
    app.run(host="0.0.0.0")


# In[ ]:




