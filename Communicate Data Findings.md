# Ford GoBike Data Exploration

## Preliminary Wrangling

This document explores a dataset GoBike is a regional public bicycle sharing system in the San Francisco, Including duration, gender, users and stations


```python
# import libraries for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# load and view datasets
df = pd.read_csv('201902-fordgobike-tripdata.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration_sec</th>
      <th>start_time</th>
      <th>end_time</th>
      <th>start_station_id</th>
      <th>start_station_name</th>
      <th>start_station_latitude</th>
      <th>start_station_longitude</th>
      <th>end_station_id</th>
      <th>end_station_name</th>
      <th>end_station_latitude</th>
      <th>end_station_longitude</th>
      <th>bike_id</th>
      <th>user_type</th>
      <th>member_birth_year</th>
      <th>member_gender</th>
      <th>bike_share_for_all_trip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52185</td>
      <td>2019-02-28 17:32:10.1450</td>
      <td>2019-03-01 08:01:55.9750</td>
      <td>21.0</td>
      <td>Montgomery St BART Station (Market St at 2nd St)</td>
      <td>37.789625</td>
      <td>-122.400811</td>
      <td>13.0</td>
      <td>Commercial St at Montgomery St</td>
      <td>37.794231</td>
      <td>-122.402923</td>
      <td>4902</td>
      <td>Customer</td>
      <td>1984.0</td>
      <td>Male</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>42521</td>
      <td>2019-02-28 18:53:21.7890</td>
      <td>2019-03-01 06:42:03.0560</td>
      <td>23.0</td>
      <td>The Embarcadero at Steuart St</td>
      <td>37.791464</td>
      <td>-122.391034</td>
      <td>81.0</td>
      <td>Berry St at 4th St</td>
      <td>37.775880</td>
      <td>-122.393170</td>
      <td>2535</td>
      <td>Customer</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61854</td>
      <td>2019-02-28 12:13:13.2180</td>
      <td>2019-03-01 05:24:08.1460</td>
      <td>86.0</td>
      <td>Market St at Dolores St</td>
      <td>37.769305</td>
      <td>-122.426826</td>
      <td>3.0</td>
      <td>Powell St BART Station (Market St at 4th St)</td>
      <td>37.786375</td>
      <td>-122.404904</td>
      <td>5905</td>
      <td>Customer</td>
      <td>1972.0</td>
      <td>Male</td>
      <td>No</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36490</td>
      <td>2019-02-28 17:54:26.0100</td>
      <td>2019-03-01 04:02:36.8420</td>
      <td>375.0</td>
      <td>Grove St at Masonic Ave</td>
      <td>37.774836</td>
      <td>-122.446546</td>
      <td>70.0</td>
      <td>Central Ave at Fell St</td>
      <td>37.773311</td>
      <td>-122.444293</td>
      <td>6638</td>
      <td>Subscriber</td>
      <td>1989.0</td>
      <td>Other</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1585</td>
      <td>2019-02-28 23:54:18.5490</td>
      <td>2019-03-01 00:20:44.0740</td>
      <td>7.0</td>
      <td>Frank H Ogawa Plaza</td>
      <td>37.804562</td>
      <td>-122.271738</td>
      <td>222.0</td>
      <td>10th Ave at E 15th St</td>
      <td>37.792714</td>
      <td>-122.248780</td>
      <td>4898</td>
      <td>Subscriber</td>
      <td>1974.0</td>
      <td>Male</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
# To discover the shape of the dataTo discover the shape of the data
df.shape
```




    (183412, 16)




```python
# To more information
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 183412 entries, 0 to 183411
    Data columns (total 16 columns):
     #   Column                   Non-Null Count   Dtype  
    ---  ------                   --------------   -----  
     0   duration_sec             183412 non-null  int64  
     1   start_time               183412 non-null  object 
     2   end_time                 183412 non-null  object 
     3   start_station_id         183215 non-null  float64
     4   start_station_name       183215 non-null  object 
     5   start_station_latitude   183412 non-null  float64
     6   start_station_longitude  183412 non-null  float64
     7   end_station_id           183215 non-null  float64
     8   end_station_name         183215 non-null  object 
     9   end_station_latitude     183412 non-null  float64
     10  end_station_longitude    183412 non-null  float64
     11  bike_id                  183412 non-null  int64  
     12  user_type                183412 non-null  object 
     13  member_birth_year        175147 non-null  float64
     14  member_gender            175147 non-null  object 
     15  bike_share_for_all_trip  183412 non-null  object 
    dtypes: float64(7), int64(2), object(7)
    memory usage: 22.4+ MB
    


```python
# To make transfers to some types of columns.

# convert start_time and end_time into datetime types.
columns = ['start_time', 'end_time']
for c in columns:
    df[c] = pd.to_datetime(df[c])
    
# Extract dayofweek, hours information from the start_time
df['start_time_dayofweek']= df['start_time'].dt.strftime('%a')
df['start_time_hour']= df['start_time'].dt.hour    
```


```python
# To duplicat
df.duplicated().sum()
```




    0




```python
# To remove null values.
df.dropna(inplace=True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration_sec</th>
      <th>start_time</th>
      <th>end_time</th>
      <th>start_station_id</th>
      <th>start_station_name</th>
      <th>start_station_latitude</th>
      <th>start_station_longitude</th>
      <th>end_station_id</th>
      <th>end_station_name</th>
      <th>end_station_latitude</th>
      <th>end_station_longitude</th>
      <th>bike_id</th>
      <th>user_type</th>
      <th>member_birth_year</th>
      <th>member_gender</th>
      <th>bike_share_for_all_trip</th>
      <th>start_time_dayofweek</th>
      <th>start_time_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52185</td>
      <td>2019-02-28 17:32:10.145</td>
      <td>2019-03-01 08:01:55.975</td>
      <td>21.0</td>
      <td>Montgomery St BART Station (Market St at 2nd St)</td>
      <td>37.789625</td>
      <td>-122.400811</td>
      <td>13.0</td>
      <td>Commercial St at Montgomery St</td>
      <td>37.794231</td>
      <td>-122.402923</td>
      <td>4902</td>
      <td>Customer</td>
      <td>1984.0</td>
      <td>Male</td>
      <td>No</td>
      <td>Thu</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>61854</td>
      <td>2019-02-28 12:13:13.218</td>
      <td>2019-03-01 05:24:08.146</td>
      <td>86.0</td>
      <td>Market St at Dolores St</td>
      <td>37.769305</td>
      <td>-122.426826</td>
      <td>3.0</td>
      <td>Powell St BART Station (Market St at 4th St)</td>
      <td>37.786375</td>
      <td>-122.404904</td>
      <td>5905</td>
      <td>Customer</td>
      <td>1972.0</td>
      <td>Male</td>
      <td>No</td>
      <td>Thu</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36490</td>
      <td>2019-02-28 17:54:26.010</td>
      <td>2019-03-01 04:02:36.842</td>
      <td>375.0</td>
      <td>Grove St at Masonic Ave</td>
      <td>37.774836</td>
      <td>-122.446546</td>
      <td>70.0</td>
      <td>Central Ave at Fell St</td>
      <td>37.773311</td>
      <td>-122.444293</td>
      <td>6638</td>
      <td>Subscriber</td>
      <td>1989.0</td>
      <td>Other</td>
      <td>No</td>
      <td>Thu</td>
      <td>17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1585</td>
      <td>2019-02-28 23:54:18.549</td>
      <td>2019-03-01 00:20:44.074</td>
      <td>7.0</td>
      <td>Frank H Ogawa Plaza</td>
      <td>37.804562</td>
      <td>-122.271738</td>
      <td>222.0</td>
      <td>10th Ave at E 15th St</td>
      <td>37.792714</td>
      <td>-122.248780</td>
      <td>4898</td>
      <td>Subscriber</td>
      <td>1974.0</td>
      <td>Male</td>
      <td>Yes</td>
      <td>Thu</td>
      <td>23</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1793</td>
      <td>2019-02-28 23:49:58.632</td>
      <td>2019-03-01 00:19:51.760</td>
      <td>93.0</td>
      <td>4th St at Mission Bay Blvd S</td>
      <td>37.770407</td>
      <td>-122.391198</td>
      <td>323.0</td>
      <td>Broadway at Kearny</td>
      <td>37.798014</td>
      <td>-122.405950</td>
      <td>5200</td>
      <td>Subscriber</td>
      <td>1959.0</td>
      <td>Male</td>
      <td>No</td>
      <td>Thu</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>




```python
# To summarize the description of data stats
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>duration_sec</th>
      <th>start_station_id</th>
      <th>start_station_latitude</th>
      <th>start_station_longitude</th>
      <th>end_station_id</th>
      <th>end_station_latitude</th>
      <th>end_station_longitude</th>
      <th>bike_id</th>
      <th>member_birth_year</th>
      <th>start_time_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>174952.000000</td>
      <td>174952.000000</td>
      <td>174952.000000</td>
      <td>174952.000000</td>
      <td>174952.000000</td>
      <td>174952.000000</td>
      <td>174952.000000</td>
      <td>174952.000000</td>
      <td>174952.000000</td>
      <td>174952.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>704.002744</td>
      <td>139.002126</td>
      <td>37.771220</td>
      <td>-122.351760</td>
      <td>136.604486</td>
      <td>37.771414</td>
      <td>-122.351335</td>
      <td>4482.587555</td>
      <td>1984.803135</td>
      <td>13.456165</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1642.204905</td>
      <td>111.648819</td>
      <td>0.100391</td>
      <td>0.117732</td>
      <td>111.335635</td>
      <td>0.100295</td>
      <td>0.117294</td>
      <td>1659.195937</td>
      <td>10.118731</td>
      <td>4.734282</td>
    </tr>
    <tr>
      <th>min</th>
      <td>61.000000</td>
      <td>3.000000</td>
      <td>37.317298</td>
      <td>-122.453704</td>
      <td>3.000000</td>
      <td>37.317298</td>
      <td>-122.453704</td>
      <td>11.000000</td>
      <td>1878.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>323.000000</td>
      <td>47.000000</td>
      <td>37.770407</td>
      <td>-122.411901</td>
      <td>44.000000</td>
      <td>37.770407</td>
      <td>-122.411647</td>
      <td>3799.000000</td>
      <td>1980.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>510.000000</td>
      <td>104.000000</td>
      <td>37.780760</td>
      <td>-122.398279</td>
      <td>101.000000</td>
      <td>37.781010</td>
      <td>-122.397437</td>
      <td>4960.000000</td>
      <td>1987.000000</td>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>789.000000</td>
      <td>239.000000</td>
      <td>37.797320</td>
      <td>-122.283093</td>
      <td>238.000000</td>
      <td>37.797673</td>
      <td>-122.286533</td>
      <td>5505.000000</td>
      <td>1992.000000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>84548.000000</td>
      <td>398.000000</td>
      <td>37.880222</td>
      <td>-121.874119</td>
      <td>398.000000</td>
      <td>37.880222</td>
      <td>-121.874119</td>
      <td>6645.000000</td>
      <td>2001.000000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>



### What is the structure of your dataset?

There are 174,952 customers in the data set with their age, gender, flights, duration and stations used.

### What is/are the main feature(s) of interest in your dataset?

I am interested in investigate duration of biking time.

### What features in the dataset do you think will help support your investigation into your feature(s) of interest?

I think features start_time, birth_year, gender, and user_type may help to support my investigation into the feature (duration).


## Univariate Exploration


```python
#Creating new columns needed to explore us

# Calculate member_age from member_birth_year.
df['member_age'] = 2021 - df['member_birth_year']

# Create a column for durations in minutes
df['duration_min'] = df['duration_sec']/60
```


```python
# To find out how long it takes to travel
binsize = 10
bins = np.arange(0, df['duration_min'].max()+binsize, binsize)

plt.figure(figsize=[8, 5])

plt.hist(data = df, x = 'duration_min', bins=bins);
```


    
![png](output_13_0.png)
    


##### It's seems that most data are below 200. Let's find out the distribution.


```python
# there's a long tail in the distribution, so let's put it on a log scale instead
log_binsize = 0.025
bins = 10 ** np.arange(0, np.log10(df['duration_min'].max())+log_binsize, log_binsize)

plt.figure(figsize=[8, 5]);
plt.hist(data = df, x = 'duration_min', bins = bins);
plt.xscale('log');
plt.xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000], [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]);
plt.xlabel('Duration (min)');
```


    
![png](output_15_0.png)
    


##### The distribution now looks closer to normal distribution. But, there is a long tail still. Let's remove the long tail.

##### Looks like it's going to take a minute to an hour and a third.


```python
# Leave record with duration_min < 100 min
df = df[df['duration_min'] <= 100]
```


```python
# To find out the age of the people used
plt.hist(data = df, x = 'member_age');
```


    
![png](output_19_0.png)
    



```python
# Leave record with duration_min < 70 min
df = df[df['member_age'] <= 60]
```


```python
binsize = 2
bins = np.arange(18, df['member_age'].max()+binsize, binsize)
plt.hist(data = df, x = 'member_age', bins = bins);
```


    
![png](output_21_0.png)
    


##### The most age-riding score is ranging 20 : 60 years.


```python
# The most requested time.
base_color = sns.color_palette()[0]
sns.countplot(data=df, x='start_time_hour', color=base_color);
```


    
![png](output_23_0.png)
    


##### It seems to be using grades throughout the day, but there's a lot of activity between 8:00 am and 5:00 pm


```python
# The gender most  requested
base_color = sns.color_palette()[0]
order = df['member_gender'].value_counts().index
sns.countplot(data=df, x='member_gender', color=base_color, order=order);
```


    
![png](output_25_0.png)
    


##### Men are more used to bike than ladies


```python
# The user_type most use
base_color = sns.color_palette()[0]
order = df['user_type'].value_counts().index
sns.countplot(data=df, x='user_type', color=base_color, order=order);
```


    
![png](output_27_0.png)
    


##### Subscribers are more request after than other customers


```python
# The most days requested 
base_color = sns.color_palette()[0]
sns.countplot(data=df, x='start_time_dayofweek', color=base_color);
```


    
![png](output_29_0.png)
    


##### It seems that Tuesday is the most riding day of the bikes.

### Bivariate Exploration


```python
all_numeric_vars = ['duration_sec', 'start_time', 'end_time', 'start_station_id', 'start_station_latitude', 
                    'start_station_longitude', 'end_station_id', 'end_station_latitude',
                    'end_station_longitude', 'bike_id', 'member_birth_year', 'member_age', 'duration_min']
numeric_vars = ['duration_min', 'member_age']
categoric_vars = ['start_time_dayofweek', 'start_time_hour', 'member_gender', 'user_type']
```


```python
# correlation plot
plt.figure(figsize = [10, 10])
sns.heatmap(df[all_numeric_vars].corr(), annot = True, fmt = '.3f', cmap = 'vlag_r', center = 0)
plt.show()
```


    
![png](output_33_0.png)
    


##### The correlation cofficients indicate there might be correlations between the following pairs of variables: (start_station_id, end_station_id), (start_station_latitude, start_station_longitude), and (end_station_latitude, end_station_longitude). However, there are no strong evidences to support these. So these might happen just by accident.


```python
g = sns.PairGrid(data = df, vars = numeric_vars, height = 4, aspect = 1.5)
g = g.map_diag(plt.hist, bins = 20);
g.map_offdiag(plt.scatter);
```


    
![png](output_35_0.png)
    



```python
# scatter plot of duration_min vs. member_age, with log transform on duration_min axis

plt.figure(figsize = [8, 6]);
plt.scatter(data = df, x = 'member_age', y = 'duration_min', alpha = 1/10);
plt.xlabel('Member Age');
plt.yscale('log');
plt.yticks([1, 2, 5, 10, 20, 50, 100], [1, 2, 5, 10, 20, 50, 100]);
plt.ylabel('Duration (min)');
```


    
![png](output_36_0.png)
    


##### Although we are still not seeing linear relationship between duration_min and member_age, the data looks distribute more even compared to the scatter plot drawn eariler.



```python
# Does gender have an effect on bikeing days?
sns.countplot(data = df, x = 'start_time_dayofweek', hue = 'member_gender', palette = 'Blues');
```


    
![png](output_38_0.png)
    


##### It seems that males are always more riding grades than females throughout the week. 


```python
# Does user have an effect on bikeing days?
sns.countplot(data = df, x = 'start_time_dayofweek', hue = 'user_type', palette = 'Blues');
```


    
![png](output_40_0.png)
    


##### It seems that the Subscriber are always more riding the score throughout the week.


```python
# The relationship between age and cycling days
base_color = sns.color_palette()[0]
sns.violinplot(data=df, x='start_time_dayofweek', y='member_age', color=base_color, inner='quartile')
plt.xticks(rotation=15);
```


    
![png](output_42_0.png)
    


##### On average, all days seem to be equal in age, but on Monday, the oldest age is recorded.   


```python
def log_trans(x, inverse = False):
    """ quick function for computing log and power operations """
    if not inverse:
        return np.log10(x)
    else:
        return np.power(10, x)

df['log_duration_min'] = df['duration_min'].apply(log_trans)
```


```python
# plot the categorical variables against duration_min and member_age again, this time
# with full data and variable transforms
fig, ax = plt.subplots(ncols = 2, nrows = 4 , figsize = [15,10])
default_color = sns.color_palette()[0];

for i in range(len(categoric_vars)):
    var = categoric_vars[i]
    sns.violinplot(data = df, x = var, y = 'log_duration_min', ax = ax[i,0], color = default_color);
    ax[i,0].set_yticks(log_trans(np.array([1, 2, 5, 10, 20, 50, 100])));
    ax[i,0].set_yticklabels([1, 2, 5, 10, 20, 50, 100]);
    sns.violinplot(data = df, x = var, y = 'member_age', ax = ax[i,1], color = default_color);
```


    
![png](output_45_0.png)
    


- The shapes of the violins in the plots on the left side (with 'log_duration_min' as y-axis) are more even compared to the shapes of the violins in the plots on the right side (with member_age as y-axis). The transformation make the violins on the left side looks more even.
- Look at the plot on 1st row, 1st column: there are more bike rides with durations close to mean duration time (10 minutes) on the weekdays compared to the bike rides on the weekends.
- Look at the plot on 3rd row, 1st column: the mean bike duration for female biker is longer than the the mean duration of the male bikers.
- Look at the plot on 4th row, 1st column: the mean bike duration for 'Customer' biker is longer than the the mean duration of the 'Subscriber' bikers.

### Multivariate Exploration


```python
def hist2dgrid(x, y, **kwargs):
    """ Quick hack for creating heat maps with seaborn's PairGrid. """
    palette = kwargs.pop('color');
    bins_x = np.arange(18, df['member_age'].max()+2, 2);
    bins_y = np.arange(0, 2, 0.1);
    plt.hist2d(x, y, bins = [bins_x, bins_y], cmap = palette, cmin = 0.5);
    plt.yticks(log_trans(np.array([1, 2, 5, 10, 20, 50, 100])), [1, 2, 5, 10, 20, 50, 100]);
```


```python
# create faceted heat maps on levels of the cut variable
g = sns.FacetGrid(data = df, col = 'start_time_dayofweek', col_wrap = 3, height = 3);
g.map(hist2dgrid, 'member_age',  'log_duration_min', color = 'inferno_r');
g.set_xlabels('Member Age');
g.set_ylabels('Duration (min)');
```


    
![png](output_49_0.png)
    


##### Bike rides on Saturday and Sunday have longer durations compared to bike rides on other weekdays.


```python
g = sns.FacetGrid(data = df, col = 'start_time_hour', col_wrap = 4, height = 3)
g.map(hist2dgrid, 'member_age', 'log_duration_min', color = 'inferno_r');
g.set_xlabels('Member Age');
g.set_ylabels('Duration (min)');
```


    
![png](output_51_0.png)
    


##### 4:00 AM has least bikers while 5:00 PM has the most bikers.


```python
g = sns.FacetGrid(data = df, col = 'member_gender', height = 3);
g.map(hist2dgrid, 'member_age', 'log_duration_min', color = 'inferno_r');
g.set_xlabels('Member Age');
g.set_ylabels('Duration (min)');
```


    
![png](output_53_0.png)
    


##### Female bikers bike longer on average compared to male bikers.


```python
g = sns.FacetGrid(data = df, col = 'user_type', height = 3);
g.map(hist2dgrid, 'member_age', 'log_duration_min', color = 'inferno_r');
g.set_xlabels('Member Age');
g.set_ylabels('Duration (min)');
```


    
![png](output_55_0.png)
    


##### 'Customer' bikers bike longer on average compared to 'Subscriber' bikers.

### Talk about some of the relationships you observed in this part of the investigation. Were there features that strengthened each other in terms of looking at your feature(s) of interest?

- The features we investigated here are pretty much indenpendant from each other. We did not observed features that strengthened each other in terms of looking at features to my interest.

### Were there any interesting or surprising interactions between features?

>- Bike rides on Saturday and Sunday have longer durations compared to bike rides on other weekdays.
>- 4:00 AM has least bikers while 5:00 PM has the most bikers.
>- Female bikers bike longer on average compared to male bikers.
>- 'Customer' bikers bike longer on average compared to 'Subscriber' bikers.


```python

```
