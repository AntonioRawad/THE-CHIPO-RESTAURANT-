#!/usr/bin/env python
# coding: utf-8

# # Ex2 - Getting and Knowing your Data

# # THE CHIPO RESTAURANT 

# ### Step 1. Import the necessary libraries

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ### Step 2. Import the dataset from this [address](https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv). 

# In[5]:


data = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', delimiter='\t')


# In[6]:


data


# In[7]:


data.describe


# # rename data to chipo 

# In[9]:


chipo = data


# # print head and first 10 raw 

# In[10]:


print(chipo.head(10))


# # Step 5. What is the number of observations in the dataset?

# In[11]:


chipo.shape


# In[12]:


print(chipo.shape[0])


# # Step 6. What is the number of columns in the dataset?

# In[13]:


chipo.shape[1]


# # Step 7. Print the name of all the columns.

# In[14]:


print(chipo.columns)


# ### Step 8. How is the dataset indexed?

# In[16]:


print(data.index)


# # Step 9. Which was the most-ordered item? 

# In[18]:


# first we group the item and their respective quantity sold by item , the we sort ascending order ,\
#we print the first line that holding the max unit sold , we proceed to print output 

item_qty = chipo.groupby('item_name')['quantity'].sum()
most_ordered_item = item_qty.sort_values(ascending=False).index[0]

print("The most-ordered item is:", most_ordered_item)


# ### Step 10. For the most-ordered item, how many items were ordered?

# In[20]:


most_ordered = chipo.groupby('item_name').quantity.sum().sort_values(ascending=False).reset_index()
most_ordered.head(1)


# # Step 11. What was the most ordered item in the choice_description column?

# In[21]:


most_ordered_choice = chipo['choice_description'].value_counts().index[0]
print(most_ordered_choice)


# # Step 12. How many items were orderd in total?

# In[22]:


total_items_ordered = chipo['quantity'].sum()
print("Total items ordered:", total_items_ordered)


# # Step 13. Turn the item price into a float

# In[29]:


print(chipo['item_price'].dtype)


# #### Step 13.b. Create a lambda function and change the type of item price

# In[33]:


chipo['item_price'] = chipo['item_price'].apply(lambda x: float(x))


# #### Step 13.c. Check the item price type

# In[34]:


print(chipo['item_price'].dtype)


# In[ ]:


### Step 14. How much was the revenue for the period in the dataset?


# In[35]:


revenue = (chipo['quantity'] * chipo['item_price']).sum()
print('Revenue for the period:', round(revenue, 2))


# ### Step 15. How many orders were made in the period?

# In[36]:


num_orders = chipo['order_id'].nunique()
print("Number of orders:", num_orders)


# In[ ]:





# In[38]:


chipo['revenue'] = chipo['quantity'] * chipo['item_price']
average_revenue = chipo.groupby('order_id')['revenue'].sum().mean()
print(f"The average revenue amount per order is ${round(average_revenue, 2)}")


# ### Step 17. How many different items are sold?

# In[39]:


num_items_sold = chipo['item_name'].nunique()
print(f"Number of different items sold: {num_items_sold}")


# ## we will try to add some data visualisation 

# In[41]:


# Group the data by order_id and count the number of items in each group
items_per_order = chipo.groupby("order_id")["quantity"].count()


# In[42]:


print(items_per_order)


# In[46]:


# Plot a histogram of the number of items per order
items_per_order = chipo.groupby("order_id")["quantity"].count()
plt.hist(items_per_order, bins=range(1, 14), align="left", rwidth=0.8)
plt.xticks(range(1, 14))
plt.xlabel("Number of Items")
plt.ylabel("Frequency")
plt.title("Number of Items per Order")


# ## revenue per order 

# In[44]:


# Group the data by order_id and calculate the total revenue per order
revenue_per_order = chipo.groupby("order_id")["item_price"].sum()

# Calculate the average revenue per order
avg_revenue_per_order = revenue_per_order.mean()

print(f"The average revenue per order is ${avg_revenue_per_order:.2f}")


# ## Distribution of prices per order:

# In[48]:


revenue_per_order = chipo.groupby("order_id")["item_price"].sum()
plt.hist(revenue_per_order, bins=range(0, 55, 5), rwidth=0.8)
plt.xticks(range(0, 55, 5))
plt.xlabel("Revenue")
plt.ylabel("Frequency")
plt.title("Revenue per Order")
plt.show()


# # LINEAR REGRESSION ANALYSIS 
# 

# ## STUDYING THE RELATIONSHIP BETWEEN ORDERED QUANTITIY AS A (INDEPENDENT VARIABLE) AND THE TOTAL REVENUE AS A (DEPENDENT VARIABLE)
# Extract the relevant columns from the data

# In[50]:


X = chipo["quantity"].values.reshape(-1, 1)  # independent variable
y = chipo["revenue"]  # dependent variable


# In[53]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[55]:


# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)


# In[56]:


# Make predictions on the testing data
y_pred = model.predict(X_test)


# In[57]:


# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean squared error:", mse)
print("R-squared score:", r2)


# In[ ]:


plot the regression line¶


# In[58]:


# Plot the regression line
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Quantity of Items Ordered')
plt.ylabel('Total Revenue')
plt.title('Linear Regression')
plt.show()

## The scatter plot shows that there is not a strong linear relationship between the quantity of items ordered and the total revenue generated. The points are scattered far from the line and there is a lot of variability in the data. This suggests that there may be other factors that are influencing the total revenue generated, and a linear regression model may not be the best way to model the relationship between the quantity of items ordered and the total revenue generated.
# # implementing  a multiple linear regression model

# In[59]:


### deciding the  independent variable 
X = chipo[['quantity', 'item_price']]
y = chipo['revenue']
# defining training variables 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # fit the multiple linear regression model to the training data:

# In[60]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# # make predictions on the test data and evaluate the model:

# In[61]:


y_pred = reg.predict(X_test)

from sklearn.metrics import r2_score

print('R-squared score:', r2_score(y_test, y_pred))


# In[ ]:





# In[63]:


# Plot the results
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x1 = X_test['quantity']
x2 = X_test['item_price']
ax.scatter(x1, x2, y_test, c='blue', marker='o', alpha=0.5)
ax.plot_trisurf(x1, x2, y_pred, color='red', alpha=0.5)
ax.set_xlabel('Quantity of Items Ordered')
ax.set_ylabel('Item Price')
ax.set_zlabel('Total Revenue')
plt.title('Multiple Linear Regression')
plt.show()

### From the 3D model, we can see that there is some relationship between the quantity of items ordered, the item price, and the total revenue, but it is not a very strong relationship. The dots are scattered quite far from the regression plane, and there are some clusters of dots that are quite far from the regression plane. This suggests that there are other factors that are influencing the total revenue besides just the quantity of items ordered and the item price. We may need to consider additional factors or explore other types of models to better predict the total revenue.
# ## new model to determin what affect the revenue is by creating a scatter plot of quantity against total revenue while holding item_price constant
## e can calculate the predicted values of total revenue for a range of quantity values while holding item_price constant at its mean or median value
# In[65]:


# Calculating the mean item_price
item_price_mean = chipo['item_price'].mean()


# In[66]:


# Create a range of quantity values
quantity_range = np.linspace(chipo['quantity'].min(), chipo['quantity'].max(), 100)


# In[70]:


# Create a new DataFrame with constant item_price and quantity range
new_data = pd.DataFrame({'quantity': quantity_range})


# In[73]:


# Predict the total revenue for the new data
total_revenue_pred = model.predict(new_data)


# In[72]:


new_data.describe()


# # plot the new model 

# In[74]:


plt.scatter(X_test['quantity'], y_test)
plt.plot(quantity_range, total_revenue_pred, color='red')
plt.xlabel('Quantity of Items Ordered')
plt.ylabel('Total Revenue')
plt.title('Linear Regression with Constant Item Price')
plt.show()


# # CONCLUSION 
### Based on the concentration of the data points around the regression line, we can infer that the linear model is a good fit for the data. Additionally, we can observe that the concentration of the data points is higher at lower quantity values, which indicates that customers tend to order fewer items per order.

Based on this analysis, we could suggest that the Chipo company might consider offering promotions or discounts for customers who order more items per order to encourage larger orders. Additionally, they could investigate the reasons behind the trend of customers ordering fewer items per order and consider strategies to increase the average order size.
# In[ ]:





# In[ ]:





# In[ ]:




