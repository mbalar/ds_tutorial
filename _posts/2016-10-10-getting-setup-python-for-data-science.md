---
layout: post
title: Getting Setup - Python for Data Science
---

*This is the 1st post in a series meant to be an accelerated learning guide, allowing you to quickly setup your environment and start playing around with data using Python (importing data, working with dataframes, referencing, filtering, cleaning, feature engineering, visualizing, machine learning, etc.)*

### Installation 

Instead of spending days installing and configuring Python and the rest of the data science stack individually, I recommend downloading the [Anaconda Python](https://www.continuum.io/downloads "Anaconda Python") distribution (version 2.7), which bundles the most useful libraries for data science. Going forward, we'll be able to use the GUI to add anything that hasn't already been included.

### Jupyter Notebook

Once installed, launch Juptyer, an interactive notebook that allows you to mainly run code, but also can include text, images, links, etc. Notebooks are easy to share and can help you craft a cohesive story around your analysis.

Whether you're on Windows or Mac, entering "jupyter notebook" in a terminal will open the Jupyter Notebook App in your web browser:

![Terminal](https://mbalar.github.io/img/terminal.jpg)

From there, navigate to a folder of your choice and click on the "New" dropdown and select "Python 2":

![App](https://mbalar.github.io/img/app.jpg)

And that's it! You should now be looking at a newly opened notebook:

![Notebook](https://mbalar.github.io/img/notebook.jpg)

### Pandas

Next, we'll use Pandas to read in some sample data to make sure everything is up-and-running correctly. Pandas is a library that sits on top of NumPY providing a nice interface and introducing the concept of dataframes, which sidesteps a lot of the complicated array manipulation we'd have to do if using NumPY alone.

Let's run the code below to create a dataframe containing information about a few tech company executives (copy/paste into cell and hit shift+enter):


```python
import pandas as pd

df = pd.DataFrame({ 'Name' : pd.Categorical(["Elon", "Sheryl", "Mark", "Marissa"]),
                    'Gender' : pd.Categorical(["M", "F", "M", "F"]),
                    'Age' : pd.Series([45, 47, 32, 41], dtype='int32'),
                    'Job' : pd.Categorical(["CEO", "COO", "CEO", "CEO"]),
                    'Company' : pd.Categorical(["Tesla", "Facebook", "Facebook", "Yahoo"])})
```

To view the dataframe we just created, simply pass its name:


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Company</th>
      <th>Gender</th>
      <th>Job</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>45</td>
      <td>Tesla</td>
      <td>M</td>
      <td>CEO</td>
      <td>Elon</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47</td>
      <td>Facebook</td>
      <td>F</td>
      <td>COO</td>
      <td>Sheryl</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>Facebook</td>
      <td>M</td>
      <td>CEO</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41</td>
      <td>Yahoo</td>
      <td>F</td>
      <td>CEO</td>
      <td>Marissa</td>
    </tr>
  </tbody>
</table>
</div>



and to reference a specific column we can append the column name:


```python
df.Company
```




    0       Tesla
    1    Facebook
    2    Facebook
    3       Yahoo
    Name: Company, dtype: category
    Categories (3, object): [Facebook, Tesla, Yahoo]



As a final check, let's compute the average age of these tech executives by gender:


```python
df.groupby('Gender').mean().Age
```




    Gender
    F    44.0
    M    38.5
    Name: Age, dtype: float64



### Wrap up

We've now setup our environment, learned about Jupyter, and explored some of the capabilities of Pandas. If you want to go more in-depth take a look at the additional resources, otherwise move onto the [next post](http://www.mitalbalar.com "Data Exploration") where we'll import and work with real data as we dive into the data exploration process.

### Additional Resources

[Jupyter Documentation](https://jupyter.readthedocs.io/en/latest/index.html "Jupyter Documentation")  
[Pandas Documentation](http://pandas.pydata.org/pandas-docs/stable/index.html "Pandas Documentation")