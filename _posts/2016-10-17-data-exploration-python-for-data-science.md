---
layout: post
title: Data Exploration - Python for Data Science
---

Excerpt
*This is the 2nd post in the Python for Data Science series*
Out-of-excerpt

*This is the 2nd post in a series meant to be an accelerated learning guide, allowing you to quickly setup your environment and start playing around with data using Python (importing data, working with dataframes, referencing, filtering, cleaning, feature engineering, visualizing, machine learning, etc.)*

### Import Data

For this and the rest of the posts in the series we'll explore survey responses of 15,000 people who are actively learning to code provided by [Free Code Camp](https://www.freecodecamp.com/2016-new-coder-survey/ "New Coder Survey"). I've extracted a subset of the data, those who have attended a coding bootcamp, which you can download in .csv format [here](https://mbalar.github.io/data/bootcamp.csv "Bootcamp Data"). Ultimately, we'll build a machine learning model to predict which respondents became software developers.

Open up a new Jupyter notebook (refer to [Getting Setup](http://mitalbalar.com/2016/10/10/getting-setup-python-for-data-science.html "Getting Setup") if you need help) and import the .csv file using the built-in functionality of Pandas (change the pathname to where you have it stored locally):


```python
import pandas as pd

df = pd.read_csv('/Users/balar/Documents/python/datasets/bootcamp.csv', header=0)
```

### Explore

That was easy! Now that we've imported the data, let's take a look at the first few rows:


```python
df.head(3)
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
    </tr>
  </tbody>
</table>
</div>



Similarly, for the last five rows we can run:


```python
df.tail() # Defaults to five when argument left blank
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>885</th>
      <td>01231cbb610953d802e1e0a591a66053</td>
      <td>1</td>
      <td>25.0</td>
      <td>female</td>
      <td>Dev Bootcamp</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>10.0</td>
      <td>10</td>
      <td>&gt;1 million</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>Bachelor's and Higher</td>
    </tr>
    <tr>
      <th>886</th>
      <td>00c4ee169a6097f617732072b5304ba3</td>
      <td>0</td>
      <td>32.0</td>
      <td>male</td>
      <td>Other</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>10.0</td>
      <td>4</td>
      <td>100,000 - 1 million</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>Bachelor's and Higher</td>
    </tr>
    <tr>
      <th>887</th>
      <td>00c33543e86585235b2556a654e33906</td>
      <td>0</td>
      <td>NaN</td>
      <td>female</td>
      <td>Hackbright Academy</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>12</td>
      <td>&gt;1 million</td>
      <td>4</td>
      <td>0</td>
      <td>3</td>
      <td>Bachelor's and Higher</td>
    </tr>
    <tr>
      <th>888</th>
      <td>02590cc39b94751bc6759aab7f0c93b6</td>
      <td>1</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>The Iron Yard</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>12.0</td>
      <td>18</td>
      <td>&gt;1 million</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>Bachelor's and Higher</td>
    </tr>
    <tr>
      <th>889</th>
      <td>01e25d485ad926725172f88917e755f3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>General Assembly</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>30.0</td>
      <td>11</td>
      <td>NaN</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



You'll notice there are some null values (NaN), which we'll handle in the next post. To see the column details, run:


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


There are 890 observations (entries) and 15 variables (columns) that exist in the dataset. We are also given the data type and number of non-null values for each variable.

Let's get a sense of what values these variables contain. Here are all of the different bootcamps that survey respondents attended:


```python
df.BootcampName.unique()
```




    array(['General Assembly', 'Flatiron School', 'App Academy', 'Other',
           'The Iron Yard', 'Prime Digital Academy', 'Turing',
           'Hackbright Academy', 'Dev Bootcamp', 'Hack Reactor'], dtype=object)



There are 10 unique values in BootcampName. How many respondents fell into each of these?


```python
df.groupby('BootcampName').size()
```




    BootcampName
    App Academy               21
    Dev Bootcamp              48
    Flatiron School           52
    General Assembly          89
    Hack Reactor              28
    Hackbright Academy        21
    Other                    535
    Prime Digital Academy     30
    The Iron Yard             39
    Turing                    27
    dtype: int64



Most were categorized as "Other" with the 2nd most attending "General Assembly" (89 or 10% of the total):


```python
df.groupby('BootcampName').size() * 100 / len(df) # Percent of total
```




    BootcampName
    App Academy               2.359551
    Dev Bootcamp              5.393258
    Flatiron School           5.842697
    General Assembly         10.000000
    Hack Reactor              3.146067
    Hackbright Academy        2.359551
    Other                    60.112360
    Prime Digital Academy     3.370787
    The Iron Yard             4.382022
    Turing                    3.033708
    dtype: float64



Try running the same set of queries for the other categorical (object) variables, such as Gender, CityPopulation, and SchoolDegree.

For all of the numerical (float64, int64) variables, we can quickly view the summary statistics using describe:


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IsSoftwareDev</th>
      <th>Age</th>
      <th>BootcampFinish</th>
      <th>BootcampLoan</th>
      <th>BootcampRecommend</th>
      <th>HoursLearning</th>
      <th>MonthsProgramming</th>
      <th>CodeEvents</th>
      <th>PodcastsListen</th>
      <th>ResourcesUse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>890.000000</td>
      <td>777.000000</td>
      <td>890.000000</td>
      <td>890.000000</td>
      <td>890.000000</td>
      <td>878.000000</td>
      <td>890.000000</td>
      <td>890.000000</td>
      <td>890.000000</td>
      <td>890.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.471910</td>
      <td>31.104247</td>
      <td>0.697753</td>
      <td>0.334831</td>
      <td>0.784270</td>
      <td>24.781321</td>
      <td>13.023596</td>
      <td>1.884270</td>
      <td>0.624719</td>
      <td>3.489888</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.499491</td>
      <td>7.860786</td>
      <td>0.459490</td>
      <td>0.472197</td>
      <td>0.411559</td>
      <td>20.147009</td>
      <td>9.693881</td>
      <td>1.613037</td>
      <td>0.977046</td>
      <td>1.977491</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>26.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>11.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>34.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>40.000000</td>
      <td>18.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>60.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>100.000000</td>
      <td>36.000000</td>
      <td>9.000000</td>
      <td>7.000000</td>
      <td>12.000000</td>
    </tr>
  </tbody>
</table>
</div>



The average age of a bootcamp attendee is 31, 70% finished the program, dedicating 25 hours a week to learning, with about 47% becoming employed as software developers. To see the same statistics for the software developer group:


```python
df[df['IsSoftwareDev'] == 1].describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IsSoftwareDev</th>
      <th>Age</th>
      <th>BootcampFinish</th>
      <th>BootcampLoan</th>
      <th>BootcampRecommend</th>
      <th>HoursLearning</th>
      <th>MonthsProgramming</th>
      <th>CodeEvents</th>
      <th>PodcastsListen</th>
      <th>ResourcesUse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>420.0</td>
      <td>368.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>409.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
      <td>420.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.0</td>
      <td>29.739130</td>
      <td>0.890476</td>
      <td>0.309524</td>
      <td>0.826190</td>
      <td>17.320293</td>
      <td>16.714286</td>
      <td>2.052381</td>
      <td>0.659524</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>6.790151</td>
      <td>0.312668</td>
      <td>0.462849</td>
      <td>0.379398</td>
      <td>17.266725</td>
      <td>9.742718</td>
      <td>1.607595</td>
      <td>0.967629</td>
      <td>1.983825</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0</td>
      <td>15.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.0</td>
      <td>25.750000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.0</td>
      <td>28.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>10.000000</td>
      <td>15.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.0</td>
      <td>32.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>24.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.0</td>
      <td>60.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>100.000000</td>
      <td>36.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
    </tr>
  </tbody>
</table>
</div>



Of those that became software developers, a higher percentage finished (90%) and recommend the bootcamp they attended, have been programming for longer, and have participated in a greater variety of coding events than the overall population. There's a lot of good information here, but it would be much easier to glean insights if we pull these measures into a visual format.

### Seaborn

Matplotlib is the main library for plotting in Python, but just as Pandas is an easy-to-use interface built on top of NumPY, Seaborn is a high-level interface built on top of matplotlib.

Unfortunately, Seaborn does not come pre-installed with Anaconda but can be added by simply entering "conda install seaborn" in a terminal window:

![Seaborn Install](https://mbalar.github.io/img/seaborn.png)

If you get prompted to update any dependent packages enter “y” to continue. Once completed, let's build our first discrete plot:


```python
%matplotlib inline
import seaborn as sns

 # Set up a grid to plot "software developer" probability by bootcamp
g = sns.PairGrid(df, y_vars="IsSoftwareDev", x_vars="BootcampName", size=4, aspect=3.25)

 # Draw a seaborn pointplot onto each Axes
g.map(sns.pointplot)
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)
```


![png](https://mbalar.github.io/img/output_29_0.png)


Clearly some bootcamps are much better at producing software developers than others. App Academy is at the higher end with around 70% of their attendees becoming software developers vs. Turing, which is hovering at 20%. 

We can also look at these discrete plots side-by-side. For Gender, SchoolDegree, and CityPopulation:


```python
g = sns.PairGrid(df, y_vars="IsSoftwareDev", x_vars=["Gender","SchoolDegree","CityPopulation"], size=4, aspect=1)

g.map(sns.pointplot)
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)
```


![png](https://mbalar.github.io/img/output_32_0.png)


The difference between genders is fairly flat, but having a Bachelor's degree or higher, as well as living in a city with more than 100k people, carries a higher incidence of software developers.

Let's dig a bit deeper into Gender and also pull in the Age variable. Faceted histograms are a great way to visualize distributions with interactions between variables:


```python
import numpy as np
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

g = sns.FacetGrid(df, row="Gender", col="IsSoftwareDev", margin_titles=True)
bins = np.linspace(0, 60, 13)
g.map(plt.hist, "Age", color="steelblue", bins=bins, lw=0)
```









![png](https://mbalar.github.io/img/output_35_1.png)


Even though the percent of software developers by gender is about the same, there were more males that attended bootcamps than females. For both genders, we see a spike at the 25-30 age range in the developer group, but less so in the non-developer group which is more evenly distributed.

Another awesome and easy to use function of Seaborn is jointplot, which creates a linear regression between two variables along with the marginal distributions. Let's take a look at Age and MonthsProgramming:


```python
sns.jointplot("Age", "MonthsProgramming", df, kind='reg');
```


![png](https://mbalar.github.io/img/output_38_0.png)


The relationship between Age and MonthsProgramming is very weak (r = 0.055). On its own this isn't very informative, but maybe if we separate the developers and non-developers:


```python
sns.lmplot("Age", "MonthsProgramming", df, col="IsSoftwareDev");
```


![png](https://mbalar.github.io/img/output_40_0.png)


The relationship is slightly stronger for developers (r = 0.13; re-run the first jointplot for the developer group to verify). Regardless of age, developers have been programming for a greater number of months. We can overlay these on top of each other to see the difference more clearly:


```python
sns.lmplot(x="Age", y="MonthsProgramming", hue="IsSoftwareDev", data=df)
```









![png](https://mbalar.github.io/img/output_42_1.png)


### Wrap up

We've uncovered some interesting insights and identified a handful of predictive variables to include in our model. If you want to try out some more visualizations, check out the documentation for Matplotlib and Seaborn below. In the [next post](http://www.mitalbalar.com "Data Munging"), we'll munge (clean and shape) our data in order to make it machine learning ready.

### Additional Reading

[Pandas Documentation](http://pandas.pydata.org/pandas-docs/stable/index.html "Pandas Documentation")  
[Matplotlib Documentation](http://matplotlib.org/contents.html "Matplotlib Documentation")  
[Seaborn Documentation](https://seaborn.github.io/index.html "Seaborn Documentation")
