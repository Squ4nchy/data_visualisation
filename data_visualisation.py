# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Make matplotlib images appear inline in code
# %matplotlib inline

# Reduce loading of auto fill for jupyter lab
# %config Completer.use_jedi = False
# -

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

covid_data = pd.read_csv('data/covid_data.csv', index_col=0)

chess_games = pd.read_csv('data/games.csv', index_col=0)

covid_curr = covid_data[(covid_data['date'] == '2020-05-19') & (~covid_data['continent'].isna()) & (covid_data['new_deaths_per_million'] > 0)]

# +
# create a figure and axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

# scatter total_cases_per_million with total_deaths_per_million
ax.scatter(covid_curr['new_cases_per_million'], covid_curr['new_deaths_per_million'])

# set a title and labels
ax.set_title('Covid Dataset', )
ax.set_xlabel('New Cases (per million)')
ax.set_ylabel('New Deaths (per million)')

# +
# create a figure and axis
fig, ax = plt.subplots()
fig.set_size_inches(15,10)

# plot each data-point
# with a randomly generated RGB colour
for i in range(len(covid_curr['location'])):
    ax.scatter(covid_curr['new_cases_per_million'][i], covid_curr['new_deaths_per_million'][i], color=np.random.rand(3,).round(1))
    
# set a title and labels
ax.set_title('Covid Dataset')
ax.set_xlabel('New Cases (per million)')
ax.set_ylabel('New Deaths (per million)')
# -

covid_to_date = covid_data[(covid_data['location'] == 'United Kingdom') & (~covid_data['continent'].isna()) & (covid_data['new_deaths_per_million'] > 0)]

# +
# get columns to plot
columns = ['new_cases_per_million', 'new_deaths_per_million', 'icu_patients_per_million', 'hosp_patients_per_million']

# create x data
x_data = covid_to_date['date'].values[0::10]

# create figure and axis
fig, ax = plt.subplots()
fig.set_size_inches(20,15)

# plot each column
for column in columns:
    ax.plot(x_data, covid_to_date[column][0::10], label=column)
    
# set title and legend
plt.xticks(rotation=-45, ha='left')
ax.set_title('COVID: Deaths, Cases, ICU and Hospital Admissions')
ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
# for tick in ax.xaxis.get_major_ticks()[1::2]:
#     tick.set_pad(15)
# -

high_rated = chess_games[(chess_games['rated'] == True) & (chess_games['black_rating'] > 2000) & (chess_games['white_rating'] > 2000)]

# +
# create figure and axis
fig, ax = plt.subplots()
fig.set_size_inches(10,8)


# plot histogram
ax.hist(high_rated['turns'])

# set title and labels
ax.set_title('High Rated Chess Games (ELO > 2000)')
ax.set_xlabel('Total Turns per Game')
ax.set_ylabel('Frequency')
# +
# create a figure and axis
fig, ax = plt.subplots()

# count the occurrence of each class
data = wine_reviews['points'].value_counts()

# get x and y data
points = data.index
frequency = data.values

# create bar chart
ax.bar(points, frequency)

# set title and labels
ax.set_title('Wine Review Scores')
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')
# -


iris.plot.scatter(x='sepal_length', y='sepal_width', title='Iris Dataset')

iris.drop(['class'], axis=1).plot.line(title='Iris Dataset')

wine_reviews['points'].plot.hist()

iris.plot.hist(subplots=True, layout=(2,2), figsize=(10,10), bins=20)

wine_reviews['points'].value_counts().sort_index().plot.bar()

wine_reviews['points'].value_counts().sort_index().plot.barh()

wine_reviews.groupby('country').price.mean().sort_values(ascending=False)[:5].plot.bar()

sns.scatterplot(x='sepal_length', y='sepal_width', data=iris)

sns.scatterplot(x='sepal_length', y='sepal_width', hue='class', data=iris)

sns.lineplot(data=iris.drop(['class'], axis=1))

sns.histplot(wine_reviews['points'], bins=10, kde=False)

sns.histplot(wine_reviews['points'], bins=10, kde=True)

sns.countplot(x=wine_reviews['points'])

df = wine_reviews[(wine_reviews['points']>=95) & (wine_reviews['price']<1000)]
sns.boxplot(x='points', y='price', data=df)

# +
# get correlation matrix
corr = iris.corr()
fig, ax = plt.subplots()

# create heatmap
im = ax.imshow(corr.values)

# set labels
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        text = ax.text(j, i, np.around(corr.iloc[i, j], decimals=2),
                       ha="center", va="center", color="black")
# -
sns.heatmap(iris.corr(), annot=True)


g = sns.FacetGrid(iris, col='class')
g = g.map(sns.kdeplot, 'sepal_length')

sns.pairplot(iris)

# +
from pandas.plotting import scatter_matrix

fig, ax = plt.subplots(figsize=(12,12))
scatter_matrix(iris, alpha=1, ax=ax)
# -


