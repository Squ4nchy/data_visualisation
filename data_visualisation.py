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

# + tags=[]
# Reduce loading of auto fill for jupyter lab
# %config Completer.use_jedi = False
# -

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# + tags=[]
# Matplotlib settings

# Make matplotlib images appear inline in code
# %matplotlib inline

d ={
    'axes.titlesize': 22,
    'axes.titleweight': 550,
    'axes.titlepad': 20,
    'axes.labelsize': 16,
    'axes.labelweight': 550,
    'axes.labelpad': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14
}

plt.rcParams.update(d)


# -

def figure_elements(size_x, size_y,
                    title=None, x_label=None, y_label=None):
    
    fig, ax = plt.subplots()
    
    # Set figure size
    fig.set_size_inches(size_x, size_y)
    
    # set a title and labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    return(fig, ax)


covid_data = pd.read_csv('data/covid_data.csv', index_col=0)

chess_games = pd.read_csv('data/games.csv', index_col=0)

covid_curr = covid_data[(covid_data['date'] == '2020-05-19') &
                        (~covid_data['continent'].isna()) &
                        (covid_data['new_deaths_per_million'] > 0)]

# +
fig, ax = figure_elements(15, 10, 'COVID Dataset', 
                          'New Cases (per Million)', 
                          'New Deaths (per Million)')

ax.scatter(covid_curr['new_cases_per_million'], covid_curr['new_deaths_per_million'])

# +
fig, ax = figure_elements(15, 10, 'COVID Dataset', 
                          'New Cases (per Million)', 
                          'New Deaths (per Million)')

# plot each data-point
# with a randomly generated RGB colour
for i in range(len(covid_curr['location'])):
    ax.scatter(covid_curr['new_cases_per_million'][i],
               covid_curr['new_deaths_per_million'][i],
               color=np.random.rand(3,).round(1))
# -

covid_to_date = covid_data[(covid_data['location'] == 'United Kingdom') &
                           (~covid_data['continent'].isna()) &
                           (covid_data['new_deaths_per_million'] > 0)]

# +
# get columns to plot
columns = ['new_cases_per_million',
           'new_deaths_per_million',
           'icu_patients_per_million',
           'hosp_patients_per_million']

# create x data
x_data = covid_to_date['date'].values[0::10]

fig, ax = figure_elements(20, 15, 'COVID: Deaths, Cases, ICU and Hospital Admissions')

# plot each column
for column in columns:
    ax.plot(x_data, covid_to_date[column][0::10], label=column)
    
# set title and legend
plt.xticks(rotation=-45, ha='left')
ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
# -

high_rated = chess_games[(chess_games['rated'] == True) &
                         (chess_games['black_rating'] > 2000) &
                         (chess_games['white_rating'] > 2000)]

# +
fig, ax = figure_elements(10, 8,
                         'High ELO Games: Total Turns to Resolution',
                         'Total Turns per Game',
                         'Frequency')

# plot histogram
ax.hist(high_rated['turns'])
# -
high_rated.columns

openings = high_rated['opening_name'].value_counts()
top_openings = openings[openings > 4].sort_index()

# +
fig, ax = figure_elements(16, 12, 'High ELO Games: Most Used Openings',
                          'Opening Name',
                          f'Times Opening Played ({len(high_rated)} Games)')

# get x and y data
points = top_openings.index
frequency = top_openings.values

for i in range(len(points)):
    ax.bar(points[i],
           frequency[i],
           color=np.random.rand(3,).round(1),
           edgecolor='black'
          )
    
ax.set_xticklabels([])
ax.set_ylim(bottom=4)
plt.legend(points, bbox_to_anchor=(1.04, 1), loc='upper left')
    
plt.show()
# -


covid_to_date.columns

# + tags=[]
high_rated.plot.scatter(x='black_rating', y='white_rating', title='High ELO: Rating Disparity')
# -

cov_columns = covid_to_date.columns
cov_columns = [x for x in columns if x not in ['total_deaths_per_million', 'total_cases_per_million', 'date']]

covid_to_date.drop(cov_columns, axis=1).plot.line(x='date', title='COVID-19: UK Deaths and Cases', rot=-45)

high_rated['turns'].plot.hist(title='High ELO Games: Turns Per Game')

chess_columns = high_rated.columns
chess_columns = [x for x in chess_columns if x not in ['turns', 'black_rating', 'white_rating']]

# How to not share x-axis?
high_rated.drop(chess_columns, axis=1).plot.hist(subplots=True, layout=(3,1), figsize=(10,10), bins=50)

not_top_openings = [x for x in high_rated['opening_name'].values if x not in top_openings]

high_rated['opening_name']

high_rated['opening_name'].drop(not_top_openings).value_counts().sort_index().plot.bar()

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
covid_ons = pd.read_excel('./data/datadownload.xlsx')


covid_ons.dropna(how='all', inplace=True)
covid_ons.dropna(how='any', axis=0, inplace=True)

covid_ons

# +
fig, ax = figure_elements(15, 10)

cov19_deaths = covid_ons[covid_ons['Year'] == 2021]['Deaths due to COVID-19']
flu_deaths = covid_ons[covid_ons['Year'] == 2021]['Deaths due to Influenza and Pneumonia']
week_no = covid_ons[covid_ons['Year'] == 2021]['Week no.']

values = [cov19_deaths, flu_deaths]

for i in range(len(values)):
    ax.plot(week_no, values[i])

plt.xticks(rotation=-45, ha='left')
ax.legend(['covid deaths', 'flu deaths'])
