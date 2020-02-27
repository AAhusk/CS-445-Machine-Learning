#!/usr/bin/env python
# coding: utf-8

# # Homework 1 - Aaron Huskerson

# # A1.1 Linear Regression with SGD

# * A1.1: *Added preliminary grading script in last cells of notebook.*

# In this assignment, you will implement three functions `train`, `use`, and `rmse` and apply them to some weather data.
# Here are the specifications for these functions, which you must satisfy.

# `model = train(X, T, learning_rate, n_epochs, verbose)`
# * `X`: is an $N$ x $D$ matrix of input data samples, one per row. $N$ is the number of samples and $D$ is the number of variable values in
# each sample.
# * `T`: is an $N$ x $K$ matrix of desired target values for each sample.  $K$ is the number of output values you want to predict for each sample.
# * `learning_rate`: is a scalar that controls the step size of each update to the weight values.
# * `n_epochs`: is the number of epochs, or passes, through all $N$ samples, to take while updating the weight values.
# * `verbose`: is True or False (default value) to control whether or not occasional text is printed to show the training progress.
# * `model`: is the returned value, which must be a dictionary with the keys `'w'`, `'Xmeans'`, `'Xstds'`, `'Tmeans'` and `'Tstds'`.
# 
# `Y = use(X, model)`
# * `X`: is an $N$ x $D$ matrix of input data samples, one per row, for which you want to predict the target values.
# * `model`: is the dictionary returned by `train`.
# * `Y`: is the returned $N$ x $K$ matrix of predicted values, one for each sample in `X`.
# 
# `result = rmse(Y, T)`
# * `Y`: is an $N$ x $K$ matrix of predictions produced by `use`.
# * `T`: is the $N$ x $K$ matrix of target values.
# * `result`: is a scalar calculated as the square root of the mean of the squared differences between each sample (row) in `Y` and `T`.

# To get you started, here are the standard imports we need.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas


# ## 60 points: 40 for train, 10 for use, 10 for rmse

# Now here is a start at defining the `train`, `use`, and `rmse`
# functions.  Fill in the correct code wherever you see `. . .` with
# one or more lines of code.

# In[2]:


def train(X, T, learning_rate, n_epochs, verbose=False):

    # Calculate means and standard deviations of each column in X and T
    Xmeans = np.mean(X, 0)
    Tmeans = np.mean(T, 0)
    Xstds = np.std(X, 0)
    Tstds = np.std(T, 0)
    
    # Use the means and standard deviations to standardize X and T
    standardizedX = (X - Xmeans) / Xstds
    standardizedT = (T - Tmeans) / Tstds
    T = standardizedT
    
    # Insert the column of constant 1's as a new initial column in X
    X = np.insert(standardizedX, 0, 1, 1)
    
    # Initialize weights to be a numpy array of the correct shape and all zeros values.
    n_attributes = X.shape[1]
    n_outputs = T.shape[1]
    
    w = np.zeros((n_attributes, n_outputs))

    n_samples = X.shape[0]
    
    for epoch in range(n_epochs):
        sqerror_sum = 0

        for n in range(n_samples):

            # Use current weight values to predict output for sample n, then
            # calculate the error, and
            # update the weight values.
            Y = X[n:n+1, :] @ w
            error = T[n:n+1, :] - Y
            w += learning_rate * X[n:n+1, :].T * error
            # Add the squared error to sqerror_sum
            sqerror_sum += error**2
            
        if verbose and (n_epochs < 11 or (epoch + 1) % (n_epochs // 10) == 0):
            rmse = np.sqrt(sqerror_sum / n_samples)
            rmse = rmse[0, 0]  # because rmse is 1x1 matrix
            print(f'Epoch {epoch + 1} RMSE {rmse:.2f}')

    return {'w': w, 'Xmeans': Xmeans, 'Xstds': Xstds,
            'Tmeans': Tmeans, 'Tstds': Tstds}



# In[3]:


def use(X, model):
    # Standardize X using Xmeans and Xstds in model
    Xmeans = model["Xmeans"]; Xstds = model["Xstds"]
    X = (X-Xmeans) / Xstds
    X = np.insert(X, 0, 1, axis=1)
    print(X.shape)
    # Predict output values using weights in model
    weights = model["w"]
    Ystd = X @ weights
    # Unstandardize the predicted output values using Tmeans and Tstds in model
    Tmeans = model["Tmeans"]; Tstds = model["Tstds"]
    Y = (Ystd * Tstds) + Tmeans
    # Return the unstandardized output values
    return Y


# In[4]:


def rmse(A, B):
    rmse1 = np.sqrt(np.mean((A - B)**2, axis=0))
    return rmse1


# Here is a simple example use of your functions to help you debug them.  Your functions must produce the same results.

# In[5]:


X = np.arange(0, 100).reshape(-1, 1)  # make X a 100 x 1 matrix
T = 0.5 + 0.3 * X + 0.005 * (X - 50) ** 2
plt.plot(X, T, '.')
plt.xlabel('X')
plt.ylabel('T');


# In[6]:


model = train(X, T, 0.01, 50, verbose=True)
model


# In[7]:


Y = use(X, model)
plt.plot(T, '.', label='T')
plt.plot(Y, '.', label='Y')
plt.legend()


# In[8]:


plt.plot(Y[:, 0], T[:, 0], 'o')
plt.xlabel('Predicted')
plt.ylabel('Actual')
a = max(min(Y[:, 0]), min(T[:, 0]))
b = min(max(Y[:, 0]), max(T[:, 0]))
plt.plot([a, b], [a, b], 'r', linewidth=3)


# ## Weather Data

# Now that your functions are working, we can apply them to some real data. We will use data
# from  [CSU's CoAgMet Station Daily Data Access](http://coagmet.colostate.edu/cgi-bin/dailydata_form.pl).
# 
# You can get the data file [here](http://www.cs.colostate.edu/~anderson/cs445/notebooks/weather.data)

# ## 5 points:
# 
# Read in the data into variable `df` using `pandas.read_csv` like we did in lecture notes.
# Missing values in this dataset are indicated by the string `'***'`.

# In[9]:


df = pandas.read_csv('weather.data', delim_whitespace=True, na_values="***")
df


# ## 5 points:
# 
# Check for missing values by showing the number of NA values, as shown in lecture notes.

# In[10]:


df.isna().sum()


# ## 5 points:
# 
# If there are missing values, remove samples that contain missing values. Prove that you
# were successful by counting the number of missing values now, which should be zero.

# In[11]:


df = df.dropna()
df.isna().sum()


# Your job is now to create a linear model that predicts the next day's average temperature (tave) from the previous day's values of
# 1. tave: average temperature
# 2. tmax: maximum temperature
# 3. tmin: minimum temperature
# 4. vp: vapor pressure
# 5. rhmax: maximum relative humidity
# 6. rhmin: minimum relative humidity
# 7. pp: precipitation
# 8. gust: wind gust speed
# 
# As a hint on how to do this, here is a list with these column names:

# In[12]:


Xnames = ['tave', 'tmax', 'tmin', 'vp', 'rhmax', 'rhmin', 'pp', 'gust']
Tnames = ['next tave']


# ## 5 points:
# 
# Now select those eight columns from `df` and convert the result to a `numpy` array.  (Easier than it sounds.)
# Then assign `X` to be all columns and all but the last row.  Assign `T` to be just the first column (tave) and all but the first sample.  So now the first row (sample) in `X` is associated with the first row (sample) in `T` which tave for the following day.

# In[13]:


data = df[Xnames].to_numpy()
X = data[:-1, :]
T = data[1:, 0:1]
T.shape, X.shape


# ## 15 points:
# 
# Use the function `train` to train a model for the `X`
# and `T` data.  Run it several times with different `learning_rate`
# and `n_epochs` values to produce decreasing errors. Use the `use`
# function and plots of `T` versus predicted `Y` values to show how
# well the model is working.  Type your observations of the plot and of the value of `rmse` to discuss how well the model succeeds.

# In[14]:


model1 = train(X, T, 0.001, 10, verbose=True)
model2 = train(X, T, 0.01, 3, verbose=True)
model3 = train(X, T, 0.00001, 20, verbose=True)
Y1 = use(X, model1)
Y2 = use(X, model2)
Y3 = use(X, model3)

plt.plot(Y1[:365,:], '.', label='Y1')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()
plt.plot(Y2[:365,:], '.', label='Y2')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()
plt.plot(Y3[:365,:], '.', label='Y3')
plt.plot(T[:365,:], '.', label='T')
plt.legend()
plt.show()


# Observations: 
# The best RMSE I could achieve was by using a learning_rate of 0.001. Note that my RMSE results were very similar when I used a learning_rate of 0.01 and 0.0001(with 20 epochs). It is interesting that with a learning rate of 0.01 and 0.001, after the first epoch, the RMSE was already small, in the 0.33-0.36 range. However, with the learning rate of 0.0001, it took 8 epochs to get down to the 0.33-0.36 range.
# 
# The model succeeds very well in all 3 cases though: looking at the plot, the predicted values in blue follow a curve very similar to the actual values in orange. Looking at all the data, in fact, the actual values plotted in orange cover most the points in blue since they are similar enough. However, there is so much total data, it is hard to see individual points; in order to see the data more clearly, I zoomed into one year of the data. Even looking at this one year of data, we see that the predicted values follow a curve very similar to the curve of the actual data.

# ## 5 points:
# 
# Print the weight values in the resulting model along with their corresponding variable names (in `Xnames`). Use the relative magnitude
# of the weight values to discuss which input variables are most significant in predicting the changes in the tave values.

# In[15]:


for column in range(len(Xnames)):
    print(Xnames[column], model1['w'][column])


# The minimum temperature of the day has the most impact on the predicted temperature of the next day.

# ## Grading and Check-in
# 
# Your notebook will be partially run and graded automatically. Test this grading process by first downloading [A1grader.zip](http://www.cs.colostate.edu/~anderson/cs445/notebooks/A1grader.zip) and extract `A1grader.py` from it. Run the code in the following cell to demonstrate an example grading session. You should see a perfect execution score of 60/60 if your functions are defined correctly. The remaining 40 points will be based on other testing and the results you obtain and your discussions.
# 
# A different, but similar, grading script will be used to grade your checked-in notebook. It will include additional tests. You should design and perform additional tests on all of your functions to be sure they run correctly before checking in your notebook.
# 
# For the grading script to run correctly, you must first name this notebook as `Lastname-A1.ipynb` with `Lastname` being your last name, and then save this notebook and check it in at the A1 assignment link in our Canvas web page.

# In[16]:


get_ipython().magic(u'run -i A1grader.py')


# ## Extra Credit: 1 point

# A typical problem when predicting the next value in a time series is
# that the best solution may be to predict the previous value.  The
# predicted value will look a lot like the input tave value shifted on
# time step later.
# 
# To do better, try predicting the change in tave from one day to the next. `T` can be assigned as

# In[17]:


data = df[Xnames].to_numpy()
X = data[:-1, :]
T = data[1:, 0:1]
Tdelta = data[1:, 0:1] -  data[:-1, 0:1]

model4 = train(X, Tdelta, 0.001, 10, True)
Y4 = use(X, model4)
NextTaves4 = X[:, 0:1] + Y4

model5 = train(X, Tdelta, 0.01, 20, True)
Y5 = use(X, model5)
NextTaves5 = X[:, 0:1] + Y5

model6 = train(X, Tdelta, 0.00001, 40, True)
Y6 = use(X, model6)
NextTaves6 = X[:, 0:1] + Y6

plt.plot(T[:365, :], '.', label='T')
plt.plot(NextTaves4[:365, :], '.',label='Y')
plt.legend(), plt.show()

plt.plot(T[:365, :], '.', label='T')
plt.plot(NextTaves5[:365, :], '.',label='Y')
plt.legend(), plt.show()

plt.plot(T[:365, :], '.', label='T')
plt.plot(NextTaves6[:365, :], '.',label='Y')
plt.legend(), plt.show()


# Based off of RMSE, this way of predicting tave is not as accurate as directly predicting tave; in this method the best RMSE I could achieve by adjusting the learning_rate and n_epochs was 0.92. In directly predicting tave, however, I was able to get an RMSE of 0.33. 
# Still, the plots for this method visually seem to be just as accurate as the method of directly predicting tave.

# Now repeat the training experiments to pick good `learning_rate` and
# `n_epochs`.  Use predicted values to produce next day tave values by
# adding the predicted values to the previous day's tave.  Use `rmse`
# to determine if this way of predicting next tave is better than
# directly predicting tave.
