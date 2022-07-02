#!/usr/bin/env python
# coding: utf-8

# # Practical example

# ## Problem

# ### Import the relevant libraries

# In[1]:


import numpy as np
from sklearn import preprocessing #will be used to standardize the inputs
import tensorflow as tf


# ## Data

# ### Extract the data from the csv

# LOAD THE CSV FILE

# In[2]:


#coma delimiter 
raw_csv_data = np.loadtxt('scaled_data', delimiter =',')

#exclude ID column (useless) and separate inputs and targets
unscaled_inputs_all = raw_csv_data[:,:-1]
targets_all = raw_csv_data[:,-1]


# In[3]:


raw_csv_data


# ### Balance the dataset

# Shuffle the dataset

# In[4]:


#take the indices from axis 0 of our scaled inputs (the target indices are taken indirectly)
shuffled_indeces = np.arange(unscaled_inputs_all.shape[0])

#shuffle them
np.random.shuffle(shuffled_indeces)

#rearrange inputs and target following shuffled indices
unscaled_inputs_all = unscaled_inputs_all[shuffled_indeces]
targets_all = targets_all[shuffled_indeces]


# Same number of 1 and 0 in the target column

# In[5]:


#count the number of 1
#declare the variable as an int to make sure it is an integer
number_of_one_targets = int(np.sum(targets_all))
number_of_one_targets


# In[6]:


#count the number of 0
#declare the variable as an int to make sure it is an integer
number_of_zeros = len(targets_all)-number_of_one_targets
number_of_zeros


# In[7]:


#keep as many 0s as we have 1s
one_targets_counter = 0
indices_to_remove =[] #must be a list or a tuple

#count the number of zeroes
for i in range(targets_all.shape[0]):
    if targets_all[i]==1:
        one_targets_counter +=1
        if one_targets_counter > number_of_zeros: #when more 0s than 1s are found
            indices_to_remove.append(i) #mark the indeces to remove in order to have the same numbers of 0s and 1s

#delete inputs and targets corresponding to the marked ibndeces
inputs_balanced = np.delete(unscaled_inputs_all, indices_to_remove, axis = 0)
targets_balanced = np.delete(targets_all, indices_to_remove, axis = 0)     


# In[8]:


number_of_one_targets = int(np.sum(targets_balanced))
number_of_one_targets


# ### Standardize the inputs

# ALREADY TAKEN CARE OF IN THE LOGISTIC REGRESSION

# ### Shuffle the data

# It is always good to shuffle the data to have a random order

# In[9]:


#take the indices from axis 0 of our scaled inputs (the target indices are taken indirectly)
shuffled_indeces = np.arange(inputs_balanced.shape[0])

#shuffle them
np.random.shuffle(shuffled_indeces)

#rearrange inputs and target following shuffled indices
shuffled_inputs = inputs_balanced[shuffled_indeces]
shuffled_targets = targets_balanced[shuffled_indeces]


# ### Split the dataset into train, validation, and test

# Determine the size of the 3 datasets

# In[10]:


#count the total number of samples
samples_count = shuffled_inputs.shape[0]

#80-10-10 split, make sure the count is an integer
train_samples_count = int(0.8*samples_count)
validation_samples_count = int(0.1*samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count #avoid rounding errors


# EXTRACT train, validation, test sets from the big dataset

# In[11]:


#first train_samples_count of the inputs/targets sets are our train set
train_inputs = shuffled_inputs[:train_samples_count] #specify the interval from which we extract the set
train_targets = shuffled_targets[:train_samples_count]

#then the following validation_samples_count of inputs/targets are our validation set
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

#the final test_samples_count are our test set
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:] 
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]


# CHECK if the train, validation, and test sets are also BALANCED

# In[12]:


print('Number of samples =', samples_count)
print('TRAIN SET: number of 1s =',np.sum(train_targets), '-- Size =', train_samples_count, '-- % of 1 =', np.sum(train_targets)/train_samples_count)
print('VALIDATION SET: number of 1s =',np.sum(validation_targets), '-- Size =', validation_samples_count, '-- % of 1 =', np.sum(validation_targets)/validation_samples_count)
print('TEST SET: number of 1s =', np.sum(test_targets), '-- Size =', test_samples_count, '-- % of 1 =',  np.sum(test_targets)/test_samples_count)


# ### Save the three datasets in *.npz

# In[13]:


#we use .npz files through np.savez('file name', labels=array_to_save, ..., ...)
np.savez('data_train', inputs= train_inputs, targets = train_targets)
np.savez('data_validation', inputs= validation_inputs, targets = validation_targets)
np.savez('data_test', inputs= test_inputs, targets = test_targets)


# # Create the Model

# LOAD THE DATA

# In[14]:


#temporary variable to store train set
npz = np.load('data_train.npz')

#extract train inputs and targets
train_inputs = npz['inputs'].astype(float) #as floats
train_targets = npz['targets'].astype(int) #as int


#do the same for validation and test set
npz = np.load('data_validation.npz')
validation_inputs = npz['inputs'].astype(float)
validation_targets = npz['targets'].astype(int) 

npz = np.load('data_test.npz')
test_inputs = npz['inputs'].astype(float) 
test_targets = npz['targets'].astype(int)

#train, validation, and test data sets are in Array form


# ### THE ACTUAL MODEL

# OUTLINE

# In[15]:


input_size = 25 #10 features
output_size = 2 #output is 0 or 1
hidden_layer_size = 100 #all hidden layers have the same size

#build the actual model
model = tf.keras.Sequential([
                            tf.keras.layers.Dense(hidden_layer_size, activation ='relu'), #1st hidden layer
                            tf.keras.layers.Dense(hidden_layer_size, activation ='tanh'), #2nd hidden layer
                            tf.keras.layers.Dense(hidden_layer_size, activation ='sigmoid'), #5th hidden layer
                            tf.keras.layers.Dense(hidden_layer_size, activation ='sigmoid'), #5th hidden layer
                            tf.keras.layers.Dense(hidden_layer_size, activation ='relu'), #5th hidden layer
                            tf.keras.layers.Dense(output_size, activation ='softmax') #model is a classifier -> softmax
                            ])


# OPTIMIZER AND LOSS FUNCTION

# In[16]:


model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# BATCH SIZE, NUMBER OF EPOCHS, AND FIT

# In[17]:


batch_size = 100
num_epochs = 50

#set up an early stopping mechanism to avoid overfitting
#the training process will stop the first time the validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=4)

model.fit(train_inputs, 
          train_targets, 
          batch_size = batch_size,
          epochs = num_epochs,
          callbacks = [early_stopping],
          validation_data = (validation_inputs, validation_targets), 
          verbose=2)


# ## Test the model

# In[18]:


test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)


# In[19]:


print('Test loss:{0:.2f}'.format(test_loss))


# In[20]:


print('Test Accuracy:{0:.2f}%'.format(test_accuracy*100))


# In[ ]:




