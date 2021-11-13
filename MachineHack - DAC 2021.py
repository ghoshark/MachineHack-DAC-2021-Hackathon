#!/usr/bin/env python
# coding: utf-8

# ### Import Python Libraries

# In[1]:


# Import the standard Python Data Science libraries for data processing and visualization
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp   

# Import libraries for statistical tests
from statsmodels.graphics.gofplots import qqplot
import warnings
warnings.filterwarnings("ignore")
import statistics
from scipy.stats import spearmanr
from scipy.stats import anderson
from scipy.stats import shapiro
from scipy.stats import kruskal 

# Import the H2O AutoML libraries
import h2o
from h2o.automl import H2OAutoML

# Import label encoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

# Define how many base models will be built for stacking ensemble
max_models=201 # 201 is the winning model count

from scipy.stats import skew
from matplotlib import pyplot
from scipy.stats import boxcox
from numpy import exp
from math import sqrt
import shap 
# For timeseries analysis
import fbprophet

# Import libraries for various types of algorithms and metrics
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn import metrics
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import a train/test set into H2O
# These libraries are needed only when running this notebook on Azure ML cloud
# from azureml.core import Workspace, Datastore, Dataset


# #### Instantiate H2O server

# In[2]:


# Attempts to start and/or connect to and H2O instance
# max_mem_size - A character string specifying the maximum size, in bytes, of the memory allocation pool to H2O. This value must a multiple of 1024 greater than 2MB. 
# Append the letter m or M to indicate megabytes, or g or G to indicate gigabytes. 
# nthreads - Number of threads in the thread pool. This relates very closely to the number of CPUs used. -1 means use all CPUs on the host (Default). A positive integer specifies the number of CPUs directly. 
# This value is only used when R starts H2O.

h2o.init(
    nthreads=-1,     # number of threads when launching a new H2O server
    max_mem_size=12  # in gigabytes
)


# ### Import Train/Test data

# In[3]:


# Import a train set into H2O
# Here, we import the training and testing datasets

train = h2o.import_file("C:\\Data_Science\\Competitions\\MachineHack-2021\\train.csv")
test = h2o.import_file("C:\\Data_Science\\Competitions\\MachineHack-2021\\test.csv")


# In[75]:


# View the top 10 records in the training dataset

train.head(10)


# In[76]:


# View the top 10 records in the testing dataset

test.head(10)


# In[4]:


#Convert H2O frame to Pandas dataframe(This is done so that data operations can be easily done)

train_as_df = h2o.as_list(train, use_pandas=True)
test_as_df = h2o.as_list(test, use_pandas=True)


# ### Exploratory Data Analysis

# ### Profile Report
# 
#     The pandas df.describe() function is great but a little basic for serious exploratory data analysis. pandas_profiling extends the pandas DataFrame with df.profile_report() for quick data analysis.

# In[5]:


#pp.ProfileReport(credit_num)
profile=pp.ProfileReport(train_as_df, minimal=False, explorative=True)
profile
profile.to_file("C:\\Data_Science\\Competitions\\MachineHack-2021\\profile_report.html")


# ### Histogram

# In[ ]:


#Sales
train_as_df['Sales'].plot.hist(bins=20)


# In[ ]:


train_as_df.head(5)


# In[78]:


#Sales
fig, axs = plt.subplots(nrows = 2, ncols=2)
fig.set_size_inches(15, 7.5)

sns.histplot(train_as_df, x="Sales", hue="Outlet_ID", element="step",stat="density", ax=axs[0][0])
sns.histplot(train_as_df, x="Sales", hue="Outlet_Year", element="step",stat="density", ax=axs[0][1])
sns.histplot(train_as_df, x="Sales", hue="Outlet_Size", element="step", stat="density",ax=axs[1][0])
sns.histplot(train_as_df, x="Sales", hue="Outlet_Location_Type", element="step", stat="density",ax=axs[1][1])
#sns.histplot(train_as_df, x="Sales", hue="Item_Type", element="step", stat="density",ax=axs[4])


# In[ ]:


# Item Weight
train_as_df['Item_W'].plot.hist(bins=20)


# In[ ]:


#Item Weight
fig, axs = plt.subplots(nrows = 2, ncols=2)
fig.set_size_inches(15, 7.5)

sns.histplot(train_as_df, x="Item_W", hue="Outlet_ID", element="step", stat="density",ax=axs[0][0])
sns.histplot(train_as_df, x="Item_W", hue="Outlet_Year", element="step", stat="density",ax=axs[0][1])
sns.histplot(train_as_df, x="Item_W", hue="Outlet_Size", element="step", stat="density",ax=axs[1][0])
sns.histplot(train_as_df, x="Item_W", hue="Outlet_Location_Type", element="step",stat="density", ax=axs[1][1])
#sns.histplot(train_as_df, x="Item_W", hue="Item_Type", element="step",stat="density", ax=axs[4])


# In[ ]:


# Item MRP
train_as_df['Item_MRP'].plot.hist(bins=20)


# In[ ]:


#Item MRP
fig, axs = plt.subplots(nrows = 2, ncols=2)
fig.set_size_inches(15, 7.5)

sns.histplot(train_as_df, x="Item_MRP", hue="Outlet_ID", element="step", stat="density",ax=axs[0][0])
sns.histplot(train_as_df, x="Item_MRP", hue="Outlet_Year", element="step", stat="density",ax=axs[0][1])
sns.histplot(train_as_df, x="Item_MRP", hue="Outlet_Size", element="step", stat="density",ax=axs[1][0])
sns.histplot(train_as_df, x="Item_MRP", hue="Outlet_Location_Type", element="step",stat="density", ax=axs[1][1])
#sns.histplot(train_as_df, x="Item_MRP", hue="Item_Type", element="step",stat="density", ax=axs[4])


# ### Box Plot

# In[ ]:


def annotate_boxplot(ax,size):
    lines = ax.get_lines()
    categories = ax.get_xticks()
    for cat in categories:
        # every 4th line at the interval of 6 is median line
        # 0 -> p25 1 -> p75 2 -> lower whisker 3 -> upper whisker 4 -> p50 5 -> upper extreme value
        y = round(lines[4+cat*6].get_ydata()[0],1) 
        ax.text(
            cat, 
            y, 
            f'{y}', 
            ha='center', 
            va='center', 
            fontweight='bold', 
            size=size,
            color='white',
            bbox=dict(facecolor='#445A64'))
    plt.tight_layout()


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
fig.set_size_inches(15, 7.5)

sns.boxplot(x="Outlet_Year",y="Sales",data=train_as_df, ax=ax[0])
ax[0].set_title('Year-wise Sales\n(Trend)', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax[0].set_xlabel('Year', fontsize = 15, fontdict=dict(weight='bold'))
ax[0].set_ylabel('Sales', fontsize = 15, fontdict=dict(weight='bold'))
annotate_boxplot(ax[0],11)

sns.boxplot(x="Outlet_Size",y="Sales",data=train_as_df, ax=ax[1])
ax[1].set_title('Sales Distribution by Outlet Size\n', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax[1].set_xlabel('Outlet Size', fontsize = 15, fontdict=dict(weight='bold'))
ax[1].set_ylabel('Sales', fontsize = 15, fontdict=dict(weight='bold'))
annotate_boxplot(ax[1],11)


# In[ ]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))
fig.set_size_inches(15, 7.5)

sns.boxplot(x="Outlet_Location_Type",y="Sales",data=train_as_df, ax=ax[0])
ax[0].set_title('Sales Distribution by Outlet Location Type', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax[0].set_xlabel('Outlet Location Type', fontsize = 15, fontdict=dict(weight='bold'))
ax[0].set_ylabel('Sales', fontsize = 15, fontdict=dict(weight='bold'))
annotate_boxplot(ax[0],11)

sns.boxplot(x="Outlet_ID",y="Sales",data=train_as_df, ax=ax[1])
ax[1].set_title('Sales Distribution by Outlet_ID\n', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax[1].set_xlabel('Outlet_ID', fontsize = 15, fontdict=dict(weight='bold'))
ax[1].set_ylabel('Sales', fontsize = 15, fontdict=dict(weight='bold'))
annotate_boxplot(ax[1],11)


# ### Time Series Analysis

# In[ ]:


# Timeseries plotting using Seaborn library

fig, ax = plt.subplots(figsize=(15, 7.5))
d = train_as_df
sns.lineplot(d['Outlet_Year'], d['Sales'], marker="o") 

ax.set_title('Combined Sales over the Years', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax.set_xlabel('Outlet Year', fontsize = 16, fontdict=dict(weight='bold'))
ax.set_ylabel('Sales Amount', fontsize = 16, fontdict=dict(weight='bold'))
plt.tick_params(axis='y', which='major', labelsize=16)
plt.tick_params(axis='x', which='major', labelsize=16)

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='large'  
)


# In[ ]:


# Timeseries plotting using Seaborn library
# Sales Distribution over the years as per Outlet ID
# Sales Distribution by Outlet ID

fig, ax = plt.subplots(figsize=(15, 7.5))
d = train_as_df
sns.lineplot(d['Outlet_Year'], d['Sales'], marker="o", hue=d["Outlet_ID"]) 

ax.set_title('Sales over the Years by Outlets', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax.set_xlabel('Outlet Year', fontsize = 16, fontdict=dict(weight='bold'))
ax.set_ylabel('Sales Amount', fontsize = 16, fontdict=dict(weight='bold'))
plt.tick_params(axis='y', which='major', labelsize=16)
plt.tick_params(axis='x', which='major', labelsize=16)


# In[ ]:


# Timeseries plotting using Seaborn library
# Sales Distribution over the years as per Outlet ID
# Sales Distribution by Outlet_Size

fig, ax = plt.subplots(figsize=(15, 7.5))
d = train_as_df
sns.lineplot(d['Outlet_Year'], d['Sales'], marker="o", hue=d["Outlet_Size"]) 

ax.set_title('Sales over the Years by Outlet Size', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax.set_xlabel('Outlet Year', fontsize = 16, fontdict=dict(weight='bold'))
ax.set_ylabel('Sales Amount', fontsize = 16, fontdict=dict(weight='bold'))
plt.tick_params(axis='y', which='major', labelsize=16)
plt.tick_params(axis='x', which='major', labelsize=16)


# In[ ]:


# Timeseries plotting using Seaborn library
# Sales Distribution over the years as per Outlet ID
# Sales Distribution by Outlet_Location_Type

fig, ax = plt.subplots(figsize=(15, 7.5))
d = train_as_df
sns.lineplot(d['Outlet_Year'], d['Sales'], marker="o", hue=d["Outlet_Location_Type"]) 

ax.set_title('Sales over the Years by Outlet_Location_Type', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax.set_xlabel('Outlet Year', fontsize = 16, fontdict=dict(weight='bold'))
ax.set_ylabel('Sales Amount', fontsize = 16, fontdict=dict(weight='bold'))
plt.tick_params(axis='y', which='major', labelsize=16)
plt.tick_params(axis='x', which='major', labelsize=16)


# In[ ]:


# Timeseries plotting using Seaborn library
# Sales Distribution over the years as per Outlet ID
# Sales Distribution by Outlet_Size

fig, ax = plt.subplots(figsize=(15, 7.5))
d = train_as_df
sns.lineplot(d['Outlet_Year'], d['Sales'], marker="o", hue=d["Item_Type"]) 

ax.set_title('Sales over the Years by Item Type', fontsize = 20, loc='center', fontdict=dict(weight='bold'))
ax.set_xlabel('Outlet Year', fontsize = 16, fontdict=dict(weight='bold'))
ax.set_ylabel('Sales Amount', fontsize = 16, fontdict=dict(weight='bold'))
plt.tick_params(axis='y', which='major', labelsize=16)
plt.tick_params(axis='x', which='major', labelsize=16)


# ### Time Series Modelling - Prophet

# In[ ]:


# Create a timeseries dataframe with Year and Sales fields
# We will do time series modelling using Prophet algorithm from Facebook

train_as_ts = train_as_df[['Outlet_Year','Sales']].copy() 
train_as_ts['Outlet_Year'] = pd.to_datetime(train_as_ts['Outlet_Year'], format='%Y') # Since field has only got year value

# Prophet requires columns in this format: ds (Date) and y (value)
train_as_ts = train_as_ts.rename(columns={'Outlet_Year': 'ds', 'Sales': 'y'})

# Build the prophet model and fit on the training data
prophet_model = fbprophet.Prophet(changepoint_prior_scale=0.15)
prophet_model.fit(train_as_ts) 


# When creating the prophet models, I set the changepoint prior to 0.15, up from the default value of 0.05. This hyperparameter is used to control how sensitive the trend is to changes, with a higher value being more sensitive and a lower value less sensitive. This value is used to combat one of the most fundamental trade-offs in machine learning: bias vs. variance

# In[ ]:


# Make a future dataframe for 5 years
forecast = prophet_model.make_future_dataframe(periods=5, freq='Y')
# Make Sales predictions for next 5 years 
df_forecast = prophet_model.predict(forecast)


# Here, under the predictions table, we are only concerned with ds, yhat_lower, yhat_upper, and yhat because these are the variables that will give us the predicted results with respect to the date specified.
# 
# yhat means the predicted output based on the input fed to the model, yhat_lower, and upper means the upper and lower value that can go based on the predicted output that is, the fluctuations that can happen

# In[ ]:


# Check the sales forecast for 5 years from 2009-2013

df_forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(5)


# In[ ]:


#Plot the output timeseries

prophet_model.plot(df_forecast)


# In[ ]:


#Checking the trends in the data

prophet_model.plot_components(df_forecast)


# Above we can see the trends with respect to year and cyclicity in a year. The first graph represents an slightly decreasing trend as we progress through the years and the latter shows a fluctuating trend in the monthly sales. 
# For most months it is steady but towards the end of the year from December to January there is some fluctuation.
# The fluctuation gains momemtum between January and February.

# ### Statistical Analysis

# In[79]:


train_as_df.head(5)


# ### Normality Distribution Tests

# ### Quantile-Quantile Plot

# In[89]:


# q-q plot
plt.figure(figsize = (15,8))
sns.set_theme(style="white")

qqplot(train_as_df['Sales'], line='s')
plt.title('Quantile-Quantile Plot')
#plt.savefig(output_dir+"probability-distribution\\qq-plot.png",bbox_inches = 'tight',pad_inches = 0)
plt.show()
plt.close()


# ### Shapiro-Wilk Test
# 
#     The Shapiro-Wilk test evaluates a data sample and quantifies how likely it is that the data was drawn from a Gaussian distribution, named for Samuel Shapiro and Martin Wilk.

# In[91]:


hypo1 = "H0 : Sample was drawn from a Gaussian distribution , Ha : Sample was not drawn from a Gaussian distribution \n" 

# H0 : Sample was drawn from a Gaussian distribution 
# Ha : Sample was not drawn from a Gaussian distribution
    
# p <= alpha: reject H0, not normal.
# p > alpha: fail to reject H0, normal.

stat, p = shapiro(train_as_df['Sales'])
title1 = 'Shapiro-Wilk Test of Normality \n'
print(title1)
print(hypo1)
print('Statistics=%.4f, p-value=%.4f \n' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    result = 'Sample looks Gaussian (fail to reject H0)'
    print("Conclusion:\n" ,result)
else:
    result = 'Sample does not look Gaussian (reject Null Hypothesis H0)'
    print("Conclusion:\n" ,result)      


# ### Anderson-Darling Test
# 
#     Anderson-Darling Test is a statistical test that can be used to evaluate whether a data sample comes from one of among many known data samples, named for Theodore Anderson and Donald Darling.

# In[93]:


hypo2 = "H0 : Sample was drawn from a Gaussian distribution , Ha : Sample was not drawn from a Gaussian distribution \n" 

# normality test
result = anderson(train_as_df['Sales'])
title2 = 'Anderson-Darling Test of Normality \n'
print(title2)
print(hypo2)
print('Statistic: %.4f \n' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('Significance Level %.4f: Critical Value %.4f, Data looks normal (fail to reject Null Hypothesis H0) \n' % (sl, cv))
	else:
		print('Significance Level %.4f: Critical Value %.4f, Data does not look normal (reject Null Hypothesis H0) \n' % (sl, cv))


# The statistical tests prove that Sales data is not normally distributed. We can try some type of transformations like log, exponential, inversions etc.
# But those conversions are not helping improve the accuracy in this case.

# ### Spearman Rank Correlation
# 
# Spearman rank correlation coefficient measures the monotonic relation between two variables. Its values range from -1 to +1 and can be interpreted as:
# 
#     +1: Perfectly monotonically increasing relationship
#     +0.8: Strong monotonically increasing relationship
#     +0.2: Weak monotonically increasing relationship
#     0: Non-monotonic relation
#     -0.2: Weak monotonically decreasing relationship
#     -0.8: Strong monotonically decreasing relationship
#     -1: Perfectly monotonically decreasing relationship
#     
# The Spearman rank-order correlation is a statistical procedure that is designed to measure the relationship between two variables on an ordinal scale of measurement.
# Pearson correlation assumes that the data we are comparing is normally distributed. When that assumption is not true, the correlation value is reflecting the true association. Spearman correlation does not assume that data is from a specific distribution, so it is a non-parametric correlation measure. 

# In[80]:


cor = train_as_df.corr(method="spearman")
print(cor)


# In[86]:


def display_correlation(df):
    r = df.corr(method="spearman")
    plt.figure(figsize=(15,7.5))
    sns.color_palette("pastel")
    heatmap = sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)
    plt.title("Spearman Correlation")
    return(r)
display_correlation(train_as_df)


# ### Kruskal-Wallis H Test
#     The Kruskal-Wallis test is a nonparametric version of the one-way analysis of variance test or ANOVA for short. A Kruskal-Wallis test is used to determine whether or not there is a statistically significant difference between the medians of three or more independent groups. It is considered to be the non-parametric equivalent of the One-Way ANOVA.
# 
#     The default assumption or the null hypothesis is that all data samples were drawn from the same distribution. Specifically, that the population medians of all groups are equal. A rejection of the null hypothesis indicates that there is enough evidence to suggest that one or more samples dominate another sample, but the test does not indicate which samples or by how much.
# 
#     A significant Kruskal–Wallis test indicates that at least one sample stochastically dominates another sample.If the results of a Kruskal-Wallis test are statistically significant, then it’s appropriate to conduct Dunn’s Test to determine exactly which groups are different.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Check Skewness or Normality Distribution

# In[ ]:


# For Item_W and Item_MRP features, lets checks their skewness. Skewness is a measure of symmetry in a distribution. Actually, it’s more correct to describe it as a measure of lack of symmetry. 
# A standard normal distribution is perfectly symmetrical and has zero skew. 

data = train_as_df['Item_MRP']
print( '\nSkewness for Item_MRP : ', skew(data))
# histogram
pyplot.hist(data)
pyplot.show()

data = train_as_df['Item_W']
print( '\nSkewness for Item_W : ', skew(data))
# histogram
pyplot.hist(data)
pyplot.show()


# ### Handle -ve Sales values

# In[ ]:


# We find there are sone -ve sales values which is not clear.
# Negative sales number might mean that these are losses , however this is not clerly degined in the problem statement
# First, we try by dropping the -ve sales records but its leads to reduced RMSE values
# Finally, we found that by turning these -ve numbers to positive , we can get a marginally better RMSE 

cols = 'Sales'
negative_sales = train_as_df[train_as_df[cols] < 0]
negative_sales
#train = train[train[cols] >= 0] 
# Convert to absolute values of sales
train_as_df['Sales'] = train_as_df['Sales'].abs()


# ### Outlier Detection

# In[ ]:


# Outlier Detection in 'Sales'
# After handling outliers in Sales, check the skewness and normality again - it should improve

data = train_as_df['Sales']
print( '\nSkewness for Sales : ', skew(data))
# histogram
pyplot.hist(data)
pyplot.show()

plt.boxplot(data, vert=False)
plt.title("Detecting outliers using Boxplot")
plt.xlabel('Sample')


# ### IQR method of outlier detection
# 
#     Calculate the interquartile range for the data.
#     Multiply the interquartile range (IQR) by 1.5 (a constant used to discern outliers).
#     Add 1.5 x (IQR) to the third quartile. Any number greater than this is a suspected outlier.
#     Subtract 1.5 x (IQR) from the first quartile. Any number less than this is a suspected outlier.

# In[ ]:


# Find the outlier datapoints in 'Sales'

# finding the 1st quartile
q1 = np.quantile(train_as_df['Sales'], 0.25)
 
# finding the 3rd quartile
q3 = np.quantile(train_as_df['Sales'], 0.75)
med = np.median(train_as_df['Sales'])
print(med)
 
# finding the iqr region
iqr = q3-q1

print('Median', med)

# finding upper and lower whiskers
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)
print(iqr, upper_bound, lower_bound)

outliers = train_as_df[(train_as_df['Sales'] <= lower_bound) | (train_as_df['Sales'] >= upper_bound)].Sales
print('The following are the outliers in the boxplot:{}'.format(outliers))


# ### Outlier Handling by Winsorization
# 
# Winsorization is a way to minimize the influence of outliers in your data by either:
#     Assigning the outlier a lower weight
#     Changing the value so that it is close to other values in the set
# 
# The data points are modified, not trimmed/removed

# In[ ]:


# Handle outliers by replacing values above/below a certain threhold with the threshold
# here, we have taken the lower and upper thresholds to be 1% and 99%
# Winsorization: Percentile based flooring and capping
removeOutlier = '0' # 0 means don't exclude outliers, this is just a flag for trying with/without outlier handling

df=train_as_df
col='Sales'
for col in df:
    #get dtype for column
    dt = df[col].dtype 
    #check if we want to handle outliers?
    if removeOutlier == '1':
        #check if it is a numbers
        if dt == 'int64' or dt == 'float64':
            df[col]=df[col].clip(upper = (df[col].quantile(0.99))) 
            df[col]=df[col].clip(lower = (df[col].quantile(0.01)))


# ### Feature Engineering

# In[ ]:


# Create a new field Item_Group based on Item_Type
# Here, we notice that Item-Type values can be grouped into some common categories of data like Drinks, Non Cosummables and Food
# Creating these new features help us better train the model in later stage

# we create a list of our IF...ELSE conditions for training data
conditions = [
    (train_as_df['Item_Type'] == 'Hard Drinks') | (train_as_df['Item_Type'] == 'Soft Drinks'),
    (train_as_df['Item_Type'] == 'Others') | (train_as_df['Item_Type'] == 'Household') | (train_as_df['Item_Type'] == 'Health and Hygiene'),
    (train_as_df['Item_Type'] == 'Baking Goods') | (train_as_df['Item_Type'] == 'Meat') | (train_as_df['Item_Type'] == 'Starchy Foods') | (train_as_df['Item_Type'] == 'Breads') | (train_as_df['Item_Type'] == 'Seafood'),
    (train_as_df['Item_Type'] == 'Fruits and Vegetables') | (train_as_df['Item_Type'] == 'Breakfast') | (train_as_df['Item_Type'] == 'Snack Foods') | (train_as_df['Item_Type'] == 'Frozen Foods') | (train_as_df['Item_Type'] == 'Canned') | (train_as_df['Item_Type'] == 'Dairy')
    ]

# create a list of the values we want to assign for each condition in train
values = ['Drinks', 'Non_Consummables', 'Food', 'Food']

# we create a list of our IF...ELSE conditions for testing data
conditions_t = [
    (test_as_df['Item_Type'] == 'Hard Drinks') | (test_as_df['Item_Type'] == 'Soft Drinks'),
    (test_as_df['Item_Type'] == 'Others') | (test_as_df['Item_Type'] == 'Household') | (test_as_df['Item_Type'] == 'Health and Hygiene'),
    (test_as_df['Item_Type'] == 'Baking Goods') | (test_as_df['Item_Type'] == 'Meat') | (test_as_df['Item_Type'] == 'Starchy Foods') | (test_as_df['Item_Type'] == 'Breads') | (test_as_df['Item_Type'] == 'Seafood'),
    (test_as_df['Item_Type'] == 'Fruits and Vegetables') | (test_as_df['Item_Type'] == 'Breakfast') | (test_as_df['Item_Type'] == 'Snack Foods') | (test_as_df['Item_Type'] == 'Frozen Foods') | (test_as_df['Item_Type'] == 'Canned') | (test_as_df['Item_Type'] == 'Dairy')
    ]

# create a list of the values we want to assign for each condition in test
values_t = ['Drinks', 'Non_Consummables', 'Food', 'Food']


# In[ ]:


# we create Item_Group based on the conditions defined above

train_as_df['Item_Group'] = np.select(conditions, values)
test_as_df['Item_Group'] = np.select(conditions_t, values_t)


# In[ ]:


#Derive the Outlet_Age column

# In the given dataset, we have a feayure called Outlet_Year but this by itself is not going to be very useful
# We know that the age of an outlet can have some impact on the sales, an older more well known outlet might have more sales than a newer one

train_as_df['Outlet_Age'] = 2021 - train_as_df['Outlet_Year']
train_as_df=train_as_df.drop(['Outlet_Year'], axis = 1)

test_as_df['Outlet_Age'] = 2021 - test_as_df['Outlet_Year']
test_as_df=test_as_df.drop(['Outlet_Year'], axis = 1)


# In[ ]:


# So far we should not get any NA values, still better to check
#Check for any missing values - train

round((train_as_df.isnull().sum() * 100/ len(train_as_df)),2).sort_values(ascending=False)


# In[ ]:


#Check for any missing values - test

round((test_as_df.isnull().sum() * 100/ len(test_as_df)),2).sort_values(ascending=False)


# In[ ]:


# Break the item id into 2 columns item code and item number
# This step is no longer applied since it does not improve the RMSE

#train['Item_Code'] = [x[:3] for x in train['Item_ID']]
#train['Item_Number'] = train['Item_ID'].str[-2:]
#train = train.drop(['Item_ID'], axis = 1)

#test['Item_Code'] = [x[:3] for x in test['Item_ID']]
#test['Item_Number'] = test['Item_ID'].str[-2:]
#test = test.drop(['Item_ID'], axis = 1)
#
#Convert Item_Number to character and sppend a prefix
#train['Item_Number'] = '__' + train['Item_Number'].astype(str)
#test['Item_Number'] = '__' + test['Item_Number'].astype(str)


# ### Categorical to Numeric Conversion
# Many machine learning algorithms cannot operate on label data directly. They require all input variables and output variables to be numeric. Since we will be applying a regression algorithm, all the features must be numeric in nature.
# We can do it by converting the existing categorical columns by applying:
# 
#     1. Label Encoding
#     Here, each unique category value is assigned an integer value.We convert the labels into a numeric form so as to convert them into the machine-readable form.
#     
#     2. One Hot Encoding 
#     For categorical variables where no ordinal relationship exists, the label encoding is not enough.
#     In fact, using this encoding and allowing the model to assume a natural ordering between categories may result in poor   performance or unexpected results (predictions halfway between categories).   
# 

# In[ ]:


# One HOT Encoding
# Define a function that will take the original dataframe and features to encode as input, 'one-hot encode' the features and then return the dataframe to calling function.

def one_hot_encode(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    original_dataframe = pd.concat([original_dataframe, dummies], axis=1)
    original_dataframe=original_dataframe.drop([feature_to_encode], axis = 1)
    return(original_dataframe)


# In[ ]:


# Label Encoding
# Define a function that will take the original dataframe and features to encode as input, 'label encode' the features and then return the dataframe to calling function.

def label_encode(original_dataframe, feature_to_encode):
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()    
    # Encode labels in column 'species'.
    original_dataframe[feature_to_encode]= label_encoder.fit_transform(original_dataframe[feature_to_encode])  
    return(original_dataframe)


# In[ ]:


#Encode the training features

train_as_df = one_hot_encode(train_as_df, 'Item_Type')
train_as_df = label_encode(train_as_df, 'Outlet_Size')
train_as_df = label_encode(train_as_df, 'Outlet_Location_Type')
train_as_df = one_hot_encode(train_as_df, 'Outlet_ID')
#train_as_df = one_hot_encode(train_as_df, 'Item_Code')
#train_as_df = one_hot_encode(train_as_df, 'Item_Number')
train_as_df = one_hot_encode(train_as_df, 'Item_Group')
train_as_df = one_hot_encode(train_as_df, 'Item_ID')


# In[ ]:


train_as_df.head(5)


# In[ ]:


#Encode the testing features

test_as_df = one_hot_encode(test_as_df, 'Item_Type')
test_as_df = label_encode(test_as_df, 'Outlet_Size')
test_as_df = label_encode(test_as_df, 'Outlet_Location_Type')
test_as_df = one_hot_encode(test_as_df, 'Outlet_ID')
#test_as_df = one_hot_encode(test_as_df, 'Item_Code')
#test_as_df = one_hot_encode(test_as_df, 'Item_Number')
test_as_df = one_hot_encode(test_as_df, 'Item_Group')
test_as_df = one_hot_encode(test_as_df, 'Item_ID')


# In[ ]:


test_as_df.head(5)


# ### Machine Learning Modelling

# In[ ]:


#Copy the train and test dataframes for the purpose of building Linear Regression model
#This is done because we will use the original dataframes for ensemble modelling at a later stage

train_as_df_LINREG = train_as_df.copy()
test_as_df_LINREG = test_as_df.copy()


# In[ ]:


#Splitting the data in 80:20 ratio

feature_columns = train_as_df_LINREG.columns.difference( ['Sales'] )
train_X, test_X, train_y, test_y = train_test_split(train_as_df_LINREG[feature_columns],
                                                  train_as_df_LINREG['Sales'],
                                                  test_size=0.20,
                                                  random_state=125)
print (len( train_X ))
print (len (train_y))
print (len( test_X))
print (len( test_y))
print (train_as_df_LINREG.shape)


# ### 1. Multivariate Linear Regression

# In[ ]:


## Linear Regression 
# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(train_X, train_y)
# Predict
y_predicted = regression_model.predict(test_X)

# model evaluation
mse = mean_squared_error(test_y, y_predicted)
r2 = r2_score(test_y, y_predicted)

# printing values
#print('Slope:' ,regression_model.coef_)
#print('Intercept:', regression_model.intercept_)
print('Root Mean Squared Error: ', sqrt(mse)) 
print('R2 Score: ', r2)

#Plot actual vs predicted y values
y_pred=grid.predict(test_X)
plt.figure(figsize=(15,7.5))
sns.distplot(y_pred,color="Blue",label="Predicted")
sns.distplot(test_y,color="Orange",label="Actual")
plt.grid(False)

#Save the figure in a file
#plt.savefig(output_dir+"regression\\actual_predicted.png",bbox_inches='tight')    
plt.show() 
plt.close()


# ### 2. Random Forest Regressor

# In[ ]:


# Model initialization
clf_rf = RandomForestRegressor(n_estimators=100) # Tried with 100, 200 etc.
# Fit the data(train the model)
clf_rf.fit(train_X, train_y)
# Predict
y_predicted = clf_rf.predict(test_X)

# model evaluation
mse = mean_squared_error(test_y, y_predicted)
r2 = r2_score(test_y, y_predicted)

# printing values
#print('Slope:' ,regression_model.coef_)
#print('Intercept:', regression_model.intercept_)
print('Root Mean Squared Error: ', sqrt(mse))
print('R2 Score: ', r2)

#Plot actual vs predicted y values
y_pred=grid.predict(test_X)
plt.figure(figsize=(15,7.5))
sns.distplot(y_pred,color="Blue",label="Predicted")
sns.distplot(test_y,color="Orange",label="Actual")
plt.grid(False)

#Save the figure in a file
#plt.savefig(output_dir+"regression\\actual_predicted.png",bbox_inches='tight')    
plt.show() 
plt.close()


# ### 3. XGBoost Regressor
# 

# In[ ]:


# Model initialization
model = xgboost.XGBRegressor() 
# Fit the data(train the model)
model.fit(train_X, train_y)
# Predict
y_predicted = model.predict(test_X)

# model evaluation
mse = mean_squared_error(test_y, y_predicted)
r2 = r2_score(test_y, y_predicted)

# printing values
#print('Slope:' ,regression_model.coef_)
#print('Intercept:', regression_model.intercept_)
print('Root Mean Squared Error: ', sqrt(mse)) 
print('R2 Score: ', r2)

#Plot actual vs predicted y values
y_pred=grid.predict(test_X)
plt.figure(figsize=(15,7.5))
sns.distplot(y_pred,color="Blue",label="Predicted")
sns.distplot(test_y,color="Orange",label="Actual")
plt.grid(False)

#Save the figure in a file
#plt.savefig(output_dir+"regression\\actual_predicted.png",bbox_inches='tight')    
plt.show() 
plt.close()


# ### Feature Importance - XGBoost

# In[ ]:


#The third method to compute feature importance in Xgboost is to use SHAP package. 
#It is model-agnostic and using the Shapley values from game theory to estimate the how does each feature contribute to the prediction.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test_X)
shap.summary_plot(shap_values, test_X, plot_type="bar")
shap.summary_plot(shap_values, test_X)


# ### Hypertune XGBoost

# In[ ]:


# Hypertune XGBoost
# To check: reduce max depth to 3, increase estimators to 1600

model = xgboost.XGBRegressor()
parameters = {'nthread':[4],
              'objective':['reg:squarederror'],
              'learning_rate': [0.01], 
              'max_depth': [5],
              'min_child_weight': [3],
              'subsample': [1],
              'colsample_bytree': [1], 
              'booster' : ['gbtree'],
              'n_estimators': [1500]} 

model = GridSearchCV(model,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=False) 


# In[72]:


# Fit the data(train the model)
model.fit(train_X, train_y)
# Predict
y_predicted = model.predict(test_X)

# model evaluation
mse = mean_squared_error(test_y, y_predicted)
r2 = r2_score(test_y, y_predicted)

# printing values
#print('Slope:' ,regression_model.coef_)
#print('Intercept:', regression_model.intercept_)
print('Root Mean Squared Error: ', sqrt(mse)) 
print('R2 Score: ', r2)

#Plot actual vs predicted y values
y_pred=grid.predict(test_X)
plt.figure(figsize=(15,7.5))
sns.distplot(y_pred,color="Blue",label="Predicted")
sns.distplot(test_y,color="Orange",label="Actual")
plt.grid(False)

#Save the figure in a file
#plt.savefig(output_dir+"regression\\actual_predicted.png",bbox_inches='tight')    
plt.show() 
plt.close()


# These ML models give us decent accuracy, but not great RMSE. We will need RMSE of around 1270 to get a rank in the leaderboard. Therefore, we will try ensemble modelling by stacking models.

# ### Stacked Ensemble Modelling - H2O

# In[ ]:


#Convert pandas dataframe back to H2O frame
# Before applying H2O automl algorithms we have to convert the pandas dataframe into H2O readable format
# We do the data processing in pandas dataframe format because its faster to do so.

train = h2o.H2OFrame(train_as_df)
test = h2o.H2OFrame(test_as_df) 


# In[ ]:


# Identify predictors and response variables
# First,  identify predictors and response variables. Since we are predicting ‘Sales’ among datapoints so it will be the response variable. 
# The remaining variables in the dataframe will form the predictor variables.

x = train.columns
y = "Sales"
x.remove(y)


# In[ ]:


# Run AutoML for certain base models (limited to 1 hour max runtime by default)
# Default number of models is 10 and 1 hour is the default runtime.
# The ‘max_models’ argument specifies the number of individuals (or “base”) models and does not include any ensemble models that can be trained separately.
# However, through multiple iterations, I found that when model count is between 150-200 its gives the best RMSE
# Also, we can run this model on a Unix /Windows machine the difference being on a Windows machine the XGBoost model is not available, so we must run it on Ubuntu server

aml = H2OAutoML(max_models=max_models, seed=1) #max_runtime_secs, max_models
aml.train(x=x, y=y, training_frame=train)   


# ### View the AutoML leaderboard
# 
# Next, we will view the AutoML Leaderboard. Since we did not specify a leaderboard_frame in the H2OAutoML.train() method for scoring and ranking the models, the AutoML leaderboard uses cross-validation metrics to rank the models.
# 
# A default performance metric for each machine learning task (binary classification, multiclass classification, regression) is specified internally and the leaderboard will be sorted by that metric. In the case of linear regression, the default ranking metric is RMSE(Root Mean Square Error). The leader model is stored at aml.leader and the leaderboard is stored at aml.leaderboard.

# In[ ]:


# Here, we see that as per RMSE ranking the best model is "StackedEnsemble_Best1000_1_AutoML_4_20211107_64351"
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)


# In[ ]:


# The leader model is stored here
# To view details about the best model, its performance metrics on cross-validated data

aml.leader


# In[ ]:


# To generate predictions on a test set, you can make predictions
# directly on the `"H2OAutoML"` object or on the leader model
# object directly
#preds = aml.predict(test)
# or
preds = aml.leader.predict(test)


# In[ ]:


preds.head(5)


# In[ ]:


#Combine the prediction with the test dataset. Then we can view the Sales prediction of each outlet

df = test.cbind(preds)
df.head(5)
# Slice cols by vector of names
res = df[:, ["predict"]]
res.head(5)
#Rename column
res.set_names(['Sales']) 


# ### Save Prediction Results
#     Save the results in a .CSV file. This is the submission file that is to be uploaded on the MachineHack website.

# In[ ]:


# Export the file
#h2o.export_file(res, path = "C:\\Data_Science\\Competitions\\MachineHack-2021\\my_submission.csv", force = True)

# Convert to Pandas dataframe
# Save as .CSV file
res_as_df = h2o.as_list(res, use_pandas=True)
res_as_df.to_csv('C:\\Data_Science\\Competitions\\MachineHack-2021\\my_submissionFile.csv', index=False)


# #### Save the model
# 
# There are two ways to save the leader model -- binary format and MOJO format. If you're taking your leader model to production, 
# then we'd suggest the MOJO format since it's optimized for production use.

# In[ ]:


h2o.save_model(aml.leader, path = "C:\\Data_Science\\Competitions\\MachineHack-2021\\h20_model_bin")


# In[ ]:


aml.leader.download_mojo(path = "C:\\Data_Science\\Competitions\\MachineHack-2021")


# ### Ensemble Exploration
# 
#     To understand how the ensemble works, let's take a peek inside the Stacked Ensemble "All Models" model. The "All Models" ensemble is an ensemble of all of the individual models in the AutoML run. This is often the top performing model on the leaderboard.

# In[ ]:


# Get model ids for all models in the AutoML Leaderboard
model_ids = list(aml.leaderboard['model_id'].as_data_frame().iloc[:,0])
# Get the "All Models" Stacked Ensemble model
se = h2o.get_model([mid for mid in model_ids if "StackedEnsemble_AllModels" in mid][0])
# Get the Stacked Ensemble metalearner model
metalearner = h2o.get_model(se.metalearner()['name'])


# Examine the variable importance of the metalearner (combiner) algorithm in the ensemble. This shows us how much each base learner is contributing to the ensemble. The AutoML Stacked Ensembles use the default metalearner algorithm (GLM with non-negative weights), so the variable importance of the metalearner is actually the standardized coefficient magnitudes of the GLM.

# In[ ]:


metalearner.coef_norm()


# In[ ]:


#We can also plot the base learner contributions to the ensemble.

get_ipython().run_line_magic('matplotlib', 'inline')
metalearner.std_coef_plot()

