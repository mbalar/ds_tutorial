---
layout: post
title: P3. Data Munging - Python for Data Science
---

*Cleaning data and engineering features in preparation to build a machine learning model.*

Now that we've imported and explored the data, we're only one step away from building our machine learning model: cleaning the data. If you haven't done so already, refer to the [Data Exploration](http://mitalbalar.com/2016/10/17/data-exploration-python-for-data-science.html "Data Exploration") post to catch up to this point. 

Let's continue using our previously opened notebook and focus our efforts on the following variables: Age, BootcampName, BootcampFinish, BootcampRecommend, MonthsProgramming, CodeEvents, PodcastsListen, and ResourcesUse. 


### Missing Values

In the previous post we had noticed that some of the variables contained "NaN" values. These are survey questions that for one reason or another the respondent decided not to answer and therefore are null or missing:


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 890 entries, 0 to 889
    Data columns (total 15 columns):
    ID                   890 non-null object
    IsSoftwareDev        890 non-null int64
    Age                  777 non-null float64
    Gender               777 non-null object
    BootcampName         890 non-null object
    BootcampFinish       890 non-null int64
    BootcampLoan         890 non-null int64
    BootcampRecommend    890 non-null int64
    HoursLearning        878 non-null float64
    MonthsProgramming    890 non-null int64
    CityPopulation       780 non-null object
    CodeEvents           890 non-null int64
    PodcastsListen       890 non-null int64
    ResourcesUse         890 non-null int64
    SchoolDegree         787 non-null object
    dtypes: float64(2), int64(8), object(5)
    memory usage: 104.4+ KB


Out of the list of variables we're concerned about, Age is the only one that has missing values. There are many ways we can go about imputing these missing values, but for simplicity let's fill them in with the median age across the entire population.

Make a copy of Age so we can retain the original values:


```python
df['AgeFill'] = df['Age']
```

Calculate and store the median age:


```python
median_age = df['Age'].median()
median_age
```




    29.0



Impute the median age for the missing values only:


```python
df.loc[(df.Age.isnull()),'AgeFill'] = median_age
```

Additionally, the fact that age was left blank could be predictive so let's create another variable for this:


```python
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
```

To verify we did all of this correctly:


```python
df[df['Age'].isnull()][['Age','AgeFill','AgeIsNull']].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>AgeFill</th>
      <th>AgeIsNull</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>NaN</td>
      <td>29.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NaN</td>
      <td>29.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>NaN</td>
      <td>29.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>29.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NaN</td>
      <td>29.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



And finally here's our dataframe with the newly added variables:


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>IsSoftwareDev</th>
      <th>Age</th>
      <th>Gender</th>
      <th>BootcampName</th>
      <th>BootcampFinish</th>
      <th>BootcampLoan</th>
      <th>BootcampRecommend</th>
      <th>HoursLearning</th>
      <th>MonthsProgramming</th>
      <th>CityPopulation</th>
      <th>CodeEvents</th>
      <th>PodcastsListen</th>
      <th>ResourcesUse</th>
      <th>SchoolDegree</th>
      <th>AgeFill</th>
      <th>AgeIsNull</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fcec97ea81a48afefd45fdaa0ba38ffb</td>
      <td>0</td>
      <td>31.0</td>
      <td>male</td>
      <td>General Assembly</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>40.0</td>
      <td>3</td>
      <td>100,000 - 1 million</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Bachelor's and Higher</td>
      <td>31.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fedcfbfd105c8f6a5242bd99355eefca</td>
      <td>1</td>
      <td>27.0</td>
      <td>male</td>
      <td>Flatiron School</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>15.0</td>
      <td>36</td>
      <td>&gt;1 million</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>Bachelor's and Higher</td>
      <td>27.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fe77569c98663547019c8cc265d77527</td>
      <td>1</td>
      <td>34.0</td>
      <td>male</td>
      <td>App Academy</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5.0</td>
      <td>24</td>
      <td>&gt;1 million</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>Bachelor's and Higher</td>
      <td>34.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ffe5c4e4932babee53c26fa49f2a409c</td>
      <td>0</td>
      <td>33.0</td>
      <td>male</td>
      <td>Other</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>18.0</td>
      <td>36</td>
      <td>100,000 - 1 million</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>Bachelor's and Higher</td>
      <td>33.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ffb4b6e4b0d1852b5c15144f3ea50f3d</td>
      <td>0</td>
      <td>21.0</td>
      <td>male</td>
      <td>Other</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10.0</td>
      <td>7</td>
      <td>&lt;100,000</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>Less than Bachelor's</td>
      <td>21.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Categorical Values

BootcampName is categorical, so we'll have to convert it to a numerical representation by creating "dummy" variables to indicate whether or not (0 or 1) the respondent attended a particular bootcamp. The Pandas get_dummies function makes this fairly quick and easy:


```python
dummies = pd.get_dummies(df['BootcampName']).rename(columns=lambda x: x.replace(' ', ''))
```

This created a new dataframe with 10 dummy variables, one for each unique BootcampName:


```python
dummies.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AppAcademy</th>
      <th>DevBootcamp</th>
      <th>FlatironSchool</th>
      <th>GeneralAssembly</th>
      <th>HackReactor</th>
      <th>HackbrightAcademy</th>
      <th>Other</th>
      <th>PrimeDigitalAcademy</th>
      <th>TheIronYard</th>
      <th>Turing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Let's concatenate this dataframe to our existing one:


```python
df = pd.concat([df,dummies], axis=1)
```

Done! Here's the updated dataframe with the newly created dummy variables:


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>IsSoftwareDev</th>
      <th>Age</th>
      <th>Gender</th>
      <th>BootcampName</th>
      <th>BootcampFinish</th>
      <th>BootcampLoan</th>
      <th>BootcampRecommend</th>
      <th>HoursLearning</th>
      <th>MonthsProgramming</th>
      <th>...</th>
      <th>AppAcademy</th>
      <th>DevBootcamp</th>
      <th>FlatironSchool</th>
      <th>GeneralAssembly</th>
      <th>HackReactor</th>
      <th>HackbrightAcademy</th>
      <th>Other</th>
      <th>PrimeDigitalAcademy</th>
      <th>TheIronYard</th>
      <th>Turing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fcec97ea81a48afefd45fdaa0ba38ffb</td>
      <td>0</td>
      <td>31.0</td>
      <td>male</td>
      <td>General Assembly</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>40.0</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fedcfbfd105c8f6a5242bd99355eefca</td>
      <td>1</td>
      <td>27.0</td>
      <td>male</td>
      <td>Flatiron School</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>15.0</td>
      <td>36</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fe77569c98663547019c8cc265d77527</td>
      <td>1</td>
      <td>34.0</td>
      <td>male</td>
      <td>App Academy</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5.0</td>
      <td>24</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ffe5c4e4932babee53c26fa49f2a409c</td>
      <td>0</td>
      <td>33.0</td>
      <td>male</td>
      <td>Other</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>18.0</td>
      <td>36</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ffb4b6e4b0d1852b5c15144f3ea50f3d</td>
      <td>0</td>
      <td>21.0</td>
      <td>male</td>
      <td>Other</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10.0</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Feature Engineering

So far we've created 12 new variables: 10 from BootcampName and 2 from Age. By doing this, we engineered features (variables) for our model. Let's create another feature, this one containing the proportion of time someone has been programming in relation to their age:


```python
df['MonthsProgramming/AgeFill'] = df.MonthsProgramming / (df.AgeFill * 12)
```

We'll see in the next post if this actually turns out to be a good predictive feature.

### Final Preparation

As a final step, let's drop the variables that we decided not to include in our model, as well as any redundant variables:


```python
df = df.drop(['ID','Age','Gender','BootcampName','BootcampLoan','HoursLearning','CityPopulation','SchoolDegree'], axis=1) 
```

This should leave us with 19 features to build from:


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IsSoftwareDev</th>
      <th>BootcampFinish</th>
      <th>BootcampRecommend</th>
      <th>MonthsProgramming</th>
      <th>CodeEvents</th>
      <th>PodcastsListen</th>
      <th>ResourcesUse</th>
      <th>AgeFill</th>
      <th>AgeIsNull</th>
      <th>AppAcademy</th>
      <th>DevBootcamp</th>
      <th>FlatironSchool</th>
      <th>GeneralAssembly</th>
      <th>HackReactor</th>
      <th>HackbrightAcademy</th>
      <th>Other</th>
      <th>PrimeDigitalAcademy</th>
      <th>TheIronYard</th>
      <th>Turing</th>
      <th>MonthsProgramming/AgeFill</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>31.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.008065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>36</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.111111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>34.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>36</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.090909</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.027778</td>
    </tr>
  </tbody>
</table>
</div>



### Wrap up

We've dealt with missing values, converted categorical variables to numeric, engineered our own features, and are now ready to apply machine learning. In the [next post](http://mitalbalar.com/2016/11/05/machine-learning-python-for-data-science.html "Machine Learning") we'll do exactly that by building a Random Forest model to predict which survey respondents became software developers.

### Additional Reading

[Pandas Documentation](http://pandas.pydata.org/pandas-docs/stable/index.html "Pandas Documentation")
