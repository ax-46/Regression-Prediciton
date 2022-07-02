#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd #handling data in tabular form
import seaborn as sns


# # The DATA

# ### LOAD the DATA

# In[2]:


raw_data = pd.read_csv('garments_worker_productivity.csv')


# In[3]:


raw_data.head()


# In[4]:


#data frame copy of the original dataset, this can be manipulated freely
df = raw_data.copy() 


# ### Explore the DataSet

# VISUALIZE THE ENTIRE DATAFRAME

# INFORMATION ON THE DATAFRAME

# In[5]:


#check for missing values (no missing values in this case, 700 values for each cathegory)
df.info()


# # PREPROCESSING

# ### DEPARTMENT

# EXTRACT A SPECIFIC COLUMN

# In[6]:


#extract the reason for absence
df['department']


# In[7]:


#return a list of all the different departments, non repeating values
#pd.unique(df['department'])

df['department'].unique()  #- this command does the same thing


# In[8]:


#"finishing " and "finishing" should be the same
#map both "finishing " to "finishing"
df['department'] = df['department'].map({'finishing ':'finishing', 'finishing':'finishing', 'sweing':'sweing'})

df.head()


# In[9]:


df['department'].unique()  #- this command does the same thing


# In[10]:


df['department'].value_counts()


# In[11]:


#turn reason for absence into a dummy variable
departments = pd.get_dummies(df['department'])
departments


# REPLACE "department" with DUMMY VARIABLES in the DATAFRAME

# In[12]:


#remove department
df = df.drop(['department'], axis = 1)

#add the new DUMMY features to the data frame
df = pd.concat([df, departments], axis=1)


# In[13]:


df


# In[14]:


#rename the new columns
df.columns.values


# In[15]:


column_names = ['date', 'quarter', 'day', 'team', 'targeted_productivity', 'smv',
       'wip', 'over_time', 'incentive', 'idle_time', 'idle_men',
       'no_of_style_change', 'no_of_workers', 'actual_productivity',
       'dept_finishing', 'dept_sweing']
df.columns = column_names
df.head()


# ### Team number

# In[16]:


#check how many team numbers
pd.unique(df['team'])


# In[17]:


#count how many for each team 
df['team'].value_counts()


# In[18]:


#get dummies from team and add them to the data frame
teams = pd.get_dummies(df['team'])
df = pd.concat([df, teams], axis=1)

df.head()


# In[19]:


#change new dummy features name
df.columns.values


# In[21]:


#replace team with the new names
df = df.drop(['team'], axis = 1)
column_names = ['date', 'quarter', 'day', 'targeted_productivity', 'smv',
       'wip', 'over_time', 'incentive', 'idle_time', 'idle_men',
       'no_of_style_change', 'no_of_workers', 'actual_productivity',
       'dept_finishing', 'dept_sweing', 'team 1', 'team 2','team 3','team 4', 'team 5', 'team 6', 'team 7', 'team 8', 'team 9', 'team 10', 'team 11',
       'team 12']
df.columns = column_names
df.head()


# **REORDER COLUMN NAMES**

# # CHECKPOINT

# CREATE A COPY OF THE CURRENT STATE OF THE DATAFRAME

# In[22]:


df_dummies = df.copy()


# ## DATE FEATURE

# In[23]:


#check the type of the first value of the "Data" series (all the other are the same)
type(df['date'][0])


# Convert it to a "timestamp" data type (useful for dates and time)

# In[24]:


#must specify the format
df['date'] = pd.to_datetime(df['date'], format = '%m/%d/%Y')
df['date']


# ### Create a month column in the data frame

# In[27]:


#visualize one value
df['date'][0]


# In[28]:


#get the month
df['date'][0].month


# In[29]:


#create a list with the month value of each row in the data frame
list_months = []

for i in range(len(df['date'])):
    list_months.append(df['date'][i].month)


# CREATE THE NEW COLUMN 

# In[30]:


df['month'] = list_months
df.head(2)


# ### day (of the Week)

# map from string to Monday = 0, Tuesday = 1, ..., Sunday = 6

# In[31]:


#visualize days of the week
df['day'].unique()  #- this command does the same thing


# In[32]:


#map names to integers, note there is no Friday
df['day'] = df['day'].map({'Thursday':3, 'Saturday':5, 'Sunday':6, 'Monday':0, 'Tuesday':1, 'Wednesday':2})

df.head()


# Get Rid of date column (not needed anymore)

# In[33]:


#drop date
df= df.drop(['date'], axis=1)
df.head()


# ## quarter

# map quarter from string to 1-4

# In[34]:


#visualize quarters to make sure there is 4 of them
df['quarter'].unique() 


# In[35]:


#map names to integers, note there is 5 of them, don' matter
df['quarter'] = df['quarter'].map({'Quarter1':1,'Quarter2':2, 'Quarter3':3, 'Quarter4':4, 'Quarter5':5})

df.head()


# ## Wip

# Replace missing values with avg (or other method)

# In[38]:


df['wip'].describe(include = 'all')


# In[58]:


df_missing_values = df.copy()


# In[59]:


df_missing_values['wip'].mean()


# In[60]:


df_missing_values['wip'].median()


# In[63]:


for i in range(len(df_missing_values['wip'])):
    if (df_missing_values['wip'][i]-df_missing_values['wip'][i]!=0):
        df_missing_values['wip'][i]= 1039.0


# In[64]:


df_missing_values.head()


# In[65]:


df_missing_values['wip'].describe(include = 'all')


# In[66]:


df_missing_values['wip'][1]


# ## TARGETED PRODUCTIVITY & ACTUAL PRODUCTIVITY

# replace with PRODUCTIVITY: productivity = 0 if tp>ap, 1 if tp<ap
# 

# In[70]:


#create a productivity list
productivity = []
for i in range(len(df_missing_values['targeted_productivity'])):
    if (df_missing_values['targeted_productivity'][i] <df_missing_values['actual_productivity'][i]):
        productivity.append(1)
    else:
        productivity.append(0)


# In[71]:


productivity


# In[73]:


#add productivity to the dataframe
df_missing_values['productivity'] = productivity
df_missing_values.head(2)


# In[74]:


#drop targeted_productivity and actual_productivity
df_missing_values = df_missing_values.drop(['targeted_productivity','actual_productivity'], axis = 1)
df_missing_values.head(2)


# ## Outliers

# EXPLORE THE DISTRIBUTION: smv

# In[84]:


#plot the distribution
sns.distplot(df_missing_values['smv'])


# In[90]:


#describe the distribution
df_missing_values['smv'].describe(include = 'all')


# In[91]:


#keep only data lesser than the 99th percentile
q = df_missing_values['smv'].quantile(0.99)
data_no_outliers = df_missing_values[df_missing_values['smv']<q]


# In[92]:


#plot the new distribution
sns.distplot(data_no_outliers['smv'])


# EXPLORE THE DISTRIBUTION: wip

# In[85]:


#plot the distribution
sns.distplot(df_missing_values['wip'])


# In[94]:


#keep only data lesser than the 99th percentile
q = data_no_outliers['wip'].quantile(0.99)
data_no_outliers = data_no_outliers[data_no_outliers['wip']<q]


# In[95]:


#plot the new distribution
sns.distplot(data_no_outliers['wip'])


# EXPLORE THE DISTRIBUTION: over_time

# In[86]:


#plot the distribution
sns.distplot(df_missing_values['over_time'])


# In[96]:


#keep only data lesser than the 99th percentile
q = data_no_outliers['over_time'].quantile(0.99)
data_no_outliers = data_no_outliers[data_no_outliers['over_time']<q]
#plot the new distribution
sns.distplot(data_no_outliers['over_time'])


# EXPLORE THE DISTRIBUTION: incentive

# In[87]:


#plot the distribution
sns.distplot(df_missing_values['incentive'])


# In[97]:


#keep only data lesser than the 99th percentile
q = data_no_outliers['incentive'].quantile(0.99)
data_no_outliers = data_no_outliers[data_no_outliers['incentive']<q]
#plot the new distribution
sns.distplot(data_no_outliers['incentive'])


# EXPLORE THE DISTRIBUTION: no_of_workers

# In[88]:


#plot the distribution
sns.distplot(df_missing_values['no_of_workers'])


# In[98]:


df_missing_values['no_of_workers'].describe(include='all')


# In[99]:


#describe the new dataframe with no outliers
data_no_outliers.describe(include='all')


# # LAST CHECKPOINT

# In[100]:


#save the final DataFrame
preprocessed_data = data_no_outliers.copy()


# In[101]:


#rearrange the columns
preprocessed_data.columns.values


# In[102]:


column_names_ordered = ['quarter', 'day','month', 'smv', 'wip', 'over_time', 'incentive',
       'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers',
       'dept_finishing', 'dept_sweing', 'team 1', 'team 2', 'team 3',
       'team 4', 'team 5', 'team 6', 'team 7', 'team 8', 'team 9',
       'team 10', 'team 11', 'team 12', 'productivity']
preprocessed_data = preprocessed_data[column_names_ordered]
preprocessed_data.head(1)


# ### Store the preprocessed DF in a .csv file

# In[103]:


preprocessed_data.to_csv('preprocessed_data', index=False)

