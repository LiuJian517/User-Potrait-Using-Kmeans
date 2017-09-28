
# coding: utf-8

# In[1]:

##import lib and data
import pandas as pd
import numpy as np
data = pd.read_csv('./business_record.csv')


# In[2]:

data.head()


# In[3]:

data.columns


# In[4]:

#choose useful feature
data = data[['user_id','start_time','end_time','current_miles','bonus']]


# In[5]:

data.head()


# In[6]:

##construct feature: F=count;
df = pd.DataFrame()
data['F'] = 1
df['F'] = data.groupby('user_id')['F'].aggregate('sum')


# In[7]:

#统计用户在时间窗口内驾驶次数
df.head()


# In[8]:

##construct feature:  M=SUM(current_miles)
df['M'] = data.groupby('user_id')['current_miles'].aggregate('sum')


# In[9]:

df.head()


# In[10]:

##construct feature: D=AVG(bonus);
df['D'] = data.groupby('user_id')['bonus'].aggregate('mean')


# In[11]:

df.head()


# In[12]:

##construct feature:  L=load_time-start_time; R=load_time-end_time; set load time = 2017.1.15
df['min_start_time'] = data.groupby("user_id")["start_time"].aggregate(np.min)
df['max_end_time'] = data.groupby("user_id")["start_time"].aggregate(np.max)


# In[13]:

df.head()


# In[14]:

##transfer type of min_start_time and max_end_time to datetime style
#copy
cnt_srs = df
cnt_srs['min_start_time'] = pd.to_datetime(cnt_srs['min_start_time'])


# In[15]:

cnt_srs['max_end_time'] = pd.to_datetime(cnt_srs['max_end_time'])
cnt_srs.head()


# In[16]:

##remove hour min sec
cnt_srs['min_start_time'] = cnt_srs.min_start_time.apply(lambda x: str(x)[:10])
cnt_srs['max_end_time'] = cnt_srs.max_end_time.apply(lambda x: str(x)[:10])


# In[17]:

cnt_srs.head()


# In[18]:

##construct load_time
cnt_srs['load_time'] = pd.datetime(2017,1,15)


# In[19]:

cnt_srs.head()


# In[20]:

cnt_srs.load_time


# In[21]:

##construct feature:  L=load_time-start_time; R=load_time-end_time; set load time = 2017.1.15
##load_time:datetime64;   min_start_time:object
##transfer object to datetime
cnt_srs['min_start_time'] = pd.to_datetime(cnt_srs['min_start_time'])
cnt_srs['max_end_time'] = pd.to_datetime(cnt_srs['max_end_time'])
cnt_srs['L'] = cnt_srs['load_time'] - cnt_srs['min_start_time']
cnt_srs['R'] = cnt_srs['load_time'] - cnt_srs['max_end_time']
cnt_srs.head()


# In[22]:

L =np.int64(cnt_srs.L)/86400000000000
R = np.int64(cnt_srs.R)/86400000000000


# In[23]:

df['L'] = L
df['R'] = R
df = df[['L','R','F','M','D']]


# In[24]:

df.head()


# In[27]:

##remove  outlier ;such as M==0(count = 341)
np.shape(df[df.M==0])


# In[28]:

df = df[df.M!=0]


# In[32]:

##using k-means
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
feature = ['L','R','F','M','D']
input_data = df[feature]
##normalization
def autoNorm(dataSet):  
    minVals = dataSet.min(0)  
    maxVals = dataSet.max(0)  
    ranges = maxVals - minVals  
    normDataSet = np.zeros(np.shape(dataSet))  
    m = dataSet.shape[0]  
    normDataSet = dataSet - np.tile(minVals, (m,1))  
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide  
    return normDataSet
normDataSet = autoNorm(input_data)


# In[33]:

normDataSet.head()


# In[34]:

#default k = 5,init = k-mean++
num_clusters = 5 
clf = KMeans(n_clusters=num_clusters,  n_init=1, verbose=1)  
clf.fit(normDataSet)  


# In[36]:

clf.labels_


# In[37]:

clf.cluster_centers_


# In[44]:

data_arrary  = np.array(normDataSet)


# In[46]:

data_arrary


# In[52]:

##plot
from matplotlib import pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data_arrary[:,0],data_arrary[:,1],15.0*clf.labels_,15*clf.labels_)"""选前两列值作为坐标"""
plt.xlabel("L")
plt.ylabel("R")
plt.show()

