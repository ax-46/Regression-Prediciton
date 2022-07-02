#!/usr/bin/env python
# coding: utf-8

# # Creating a logistic regression to predict productivity

# ## Import the relevant libraries

# In[1]:


# import the relevant libraries
import pandas as pd
import numpy as np


# ## Load the data

# In[2]:


# load the preprocessed CSV data
data_preprocessed = pd.read_csv('preprocessed_data')


# In[3]:


# eyeball the data
data_preprocessed.head()


# ## A comment on the targets

# In[4]:


# check if dataset is balanced (what % of targets are 1s)
data_preprocessed['productivity'].sum() / data_preprocessed['productivity'].shape[0]


# ## BALANCE THE DATAset

# ## Select the inputs for the regression

# In[5]:


data_preprocessed.shape


# In[6]:


# Selects all rows and all columns but the last one (basically the same operation)
data_preprocessed.iloc[:,:-1]


# In[7]:


# Create a variable that will contain the inputs (everything without the targets)
unscaled_inputs = data_preprocessed.iloc[:,:-1]

#checkpoint before backward elimination
check_point_inputs = unscaled_inputs.copy()


# ## Standardize the data

# In[8]:


# standardize the inputs
from sklearn.preprocessing import StandardScaler

# define scaler as an object
scaler = StandardScaler()


# In[9]:


#create a df with only non dummy features
df_no_dummies = unscaled_inputs.copy()
df_no_dummies = df_no_dummies.drop(['dept_finishing', 'dept_sweing', 'team 1', 'team 2','team 3','team 4', 'team 5', 'team 6', 'team 7', 'team 8', 'team 9', 'team 10', 'team 11',
       'team 12'], axis=1)
df_no_dummies.head()


# In[10]:


#scale the non dummy variables
scaler.fit(df_no_dummies)
inputs_scaled = scaler.transform(df_no_dummies)


# In[11]:


inputs_scaled


# In[12]:


inputs_scaled.shape


# SUBSTITUTE SCALED VALUES IN THE INPUTS DATA FRAME

# In[13]:


#map one by one the input columns that have been scaled

for i in range(7):
    for j in range (len(unscaled_inputs['wip'])):
        unscaled_inputs.iloc[j,i] = inputs_scaled[j][i]

unscaled_inputs.head()


# In[14]:


scaled_data = pd.concat([unscaled_inputs, data_preprocessed['productivity']], axis=1)
scaled_data.head(2)


# In[15]:


#saved the scaled inputs in a file to use the newral network
scaled_data.to_csv('scaled_data', index=False)


# ## Split the data into train & test and shuffle

# ### Import the relevant module

# In[16]:


# import train_test_split so we can split our data into train and test
from sklearn.model_selection import train_test_split


# ### Split

# In[17]:


#sklearn works with ndarray, transform inputs dataframe into nd array
scaled_inputs = unscaled_inputs.to_numpy()
scaled_inputs


# In[18]:


#same with target
targets = data_preprocessed['productivity'].to_numpy()
targets


# In[19]:


# declare 4 variables for the split
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, 
                                                                            test_size = 0.2, random_state = 46)


# In[20]:


# check the shape of the train inputs and targets
print (x_train.shape, y_train.shape)


# In[21]:


# check the shape of the test inputs and targets
print (x_test.shape, y_test.shape)


# ## Logistic regression with sklearn

# In[22]:


# import the LogReg model from sklearn
from sklearn.linear_model import LogisticRegression

# import the 'metrics' module, which includes important metrics we may want to use
from sklearn import metrics


# ### Training the model

# In[23]:


# create a logistic regression object
reg = LogisticRegression()


# In[24]:


# fit our train inputs
# that is basically the whole training part of the machine learning
reg.fit(x_train,y_train)


# In[25]:


# assess the train accuracy of the model
reg.score(x_train,y_train)


# ### Manually check the accuracy

# In[26]:


# find the model outputs according to our model
model_outputs = reg.predict(x_train)
model_outputs


# In[27]:


# compare them with the targets
y_train


# In[28]:


# ACTUALLY compare the two variables
model_outputs == y_train


# In[29]:


# find out in how many instances we predicted correctly
np.sum((model_outputs==y_train))


# In[30]:


# get the total number of instances
model_outputs.shape[0]


# In[31]:


# calculate the accuracy of the model
np.sum((model_outputs==y_train)) / model_outputs.shape[0]


# ### Finding the intercept and coefficients

# In[32]:


# get the intercept (bias) of our model
reg.intercept_


# In[33]:


# get the coefficients (weights) of our model
reg.coef_


# In[34]:


# check what were the names of our columns
unscaled_inputs.columns.values


# In[35]:


# save the names of the columns in an ad-hoc variable
feature_name = unscaled_inputs.columns.values


# In[36]:


# use the coefficients from this table (they will be exported later and will be used in Tableau)
# transpose the model coefficients (model.coef_) and throws them into a df (a vertical organization, so that they can be
# multiplied by certain matrices later) 
summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)

# add the coefficient values to the summary table
summary_table['Coefficient'] = np.transpose(reg.coef_)

# display the summary table
summary_table


# In[37]:


# do a little Python trick to move the intercept to the top of the summary table
# move all indices by 1
summary_table.index = summary_table.index + 1

# add the intercept at index 0
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# sort the df by index
summary_table = summary_table.sort_index()
summary_table


# ## Interpreting the coefficients

# In[38]:


# create a new Series called: 'Odds ratio' which will show the.. odds ratio of each feature
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)


# In[39]:


# display the df
summary_table


# In[40]:


# sort the table according to odds ratio
# note that by default, the sort_values method sorts values by 'ascending'
summary_table.sort_values('Odds_ratio', ascending=False)


# # TEST the Model

# In[42]:


#test the accuracy of the model on the test sets
reg.score(x_test, y_test)


# # SAVE THE MODEL

# Saving the model = saving the reg object

# Pickle is the standard Python tool for serialization and deserialization. In simple words, pickling means: converting a Python object (no matter what) into a string of characters. Logically, unpickling is about converting a string of characters (that has been pickled) into a Python object.

# The second step of the deployment is about creating a mechanism to load the saved model and make predictions
