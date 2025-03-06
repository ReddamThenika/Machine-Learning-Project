#!/usr/bin/env python
# coding: utf-8

# In[117]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# In[118]:


train_data = pd.read_excel(r"Data_Train.xlsx")


# In[119]:


pd.set_option('display.max_columns',None)


# In[120]:


train_data.head()


# In[121]:


train_data.info()


# In[122]:


train_data["Duration"].value_counts()


# In[123]:


train_data.dropna(inplace = True)


# In[124]:


train_data.isnull().sum()


# In[125]:


train_data["Journey_day"] = pd.to_datetime(train_data["Date_of_Journey"], format="%d/%m/%Y").dt.day


# In[126]:


train_data["Journey_month"] = pd.to_datetime(train_data["Date_of_Journey"], format="%d/%m/%Y").dt.month


# In[127]:


train_data.head()


# In[128]:


train_data.drop(["Date_of_Journey"], axis=1, inplace=True)


# In[129]:


train_data


# In[130]:


train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour


# In[131]:


train_data["Dep_min"] =  pd.to_datetime(train_data["Dep_Time"]).dt.minute


# In[132]:


train_data.drop(["Dep_Time"], axis= 1, inplace= True)


# In[133]:


train_data.head()


# In[134]:


train_data["Arrival_hour"]= pd.to_datetime(train_data.Arrival_Time).dt.hour


# In[135]:


train_data["Arrival_min"]= pd.to_datetime(train_data.Arrival_Time).dt.minute


# In[136]:


train_data.drop(["Arrival_Time"], axis = 1, inplace = True)


# In[137]:


train_data.head()


# In[139]:


duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration


# In[140]:


train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins



# In[141]:


train_data.drop(["Duration"], axis = 1,inplace = True)


# In[142]:


train_data.head()

train_data.head()
# In[143]:


train_data["Airline"].value_counts()


# In[144]:


sns.catplot(y = "Price", x = "Airline", data = train_data.sort_values("Price", ascending = False), kind="boxen", height = 6, aspect = 3)
plt.show()


# In[145]:


Airline = train_data[["Airline"]]

Airline = pd.get_dummies(Airline, drop_first= True)

Airline.head()


# In[ ]:


Source = train_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

Source.head()


# In[146]:


train_data["Destination"].value_counts()


# In[165]:


Destination = train_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

Destination.head()


# In[148]:


train_data["Route"]


# In[149]:


train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)


# In[150]:


train_data["Total_Stops"].value_counts()


# In[151]:


train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)


# In[152]:


train_data.head()


# In[153]:


data_train = pd.concat([train_data, Airline, Source, Destination], axis = 1)


# In[154]:


data_train.head()


# In[156]:


data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)


# In[157]:


data_train.head()


# In[158]:


data_train.shape


# In[159]:


data_train.shape


# In[160]:


data_train.columns


# In[161]:


X = data_train.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[167]:


y = data_train.iloc[:, 1]
y.head()


# In[183]:


plt.figure(figsize=(18, 18))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")

plt.show()


# In[173]:


from sklearn.ensemble import ExtraTreesRegressor
selection = ExtraTreesRegressor()
selection.fit(X, y)


# In[174]:


print(selection.feature_importances_)


# In[175]:


plt.figure(figsize = (12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='bar')
plt.show()


# In[176]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 50)


# In[177]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)


# In[180]:


y_pred = reg_rf.predict(X_test)


# In[181]:


y_pred = reg_rf.predict(X_test)


# In[182]:


reg_rf.score(X_test, y_test)


# In[ ]:




