# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:02:39 2020

@author: Vatsal Shah
logic and code by kaggle user Andrew Lukyanenko
"""


import pandas as pd
pd.set_option('max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, KFold
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))
import gc
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

dfTrain = pd.read_csv("train.csv")

dfTest = pd.read_csv("test.csv")

shapeTrain = dfTrain.shape
shapeTest = dfTest.shape

print(shapeTrain)
print(shapeTest)

#Initail data exploration 
dfTrain.head()
dfTrain['belongs_to_collection']

#Cheacking for missing values in both test and train
dfTrain.isna().sum()
dfTest.isna().sum()

#totalTrain = dfTrain.isna().sum().sort_values(ascending = False)
#percentTrain = (dfTrain.isna().sum()/dfTrain.isna().count()).sort_values(ascending= False)
#missingTrain = pd.concat([totalTrain, percentTrain], axis=1, keys=['Total','Percent'])

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
import ast
def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df

dfTrain = text_to_dict(dfTrain)
dfTest = text_to_dict(dfTest)

# Data cleaning and simplification

# Belongs to collection
for i, e in enumerate(dfTrain['belongs_to_collection'][:5]):
	print(i, e)
	
dfTrain['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0).value_counts()

dfTrain['collection_name'] = dfTrain['belongs_to_collection'].apply(lambda x: x[0]['name']
																	if x != {} else 0)

dfTrain['has_collection'] = dfTrain['belongs_to_collection'].apply(lambda x: len(x)
																   if x != {} else 0)


dfTest['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0).value_counts()

dfTest['collection_name'] = dfTest['belongs_to_collection'].apply(lambda x: x[0]['name']
																  if x != {} else 0)
dfTest['has_collection'] = dfTest['belongs_to_collection'].apply(lambda x: len(x)
																 if x != {} else 0)

dfTrain = dfTrain.drop(['belongs_to_collection'], axis = 1)
dfTest = dfTest.drop(['belongs_to_collection'], axis = 1)

#Genres
dfTrain['genres'].head(10)

for i, e in enumerate(dfTrain['genres'][:5]):
	print(i, e)

# Movies have numbers of different genres	
print('Number of genres in film')
dfTrain['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()

listOfGenres = list(dfTrain['genres'].apply(lambda x: [i['name']for i in x]
											if x != {} else []).values)
# Creating a wordcloud of the genres
plt.figure(figsize = (12,8))
text = ' '.join([i for j in listOfGenres for i in j])
wordcloud = WordCloud(max_font_size= None, background_color= 'white', collocations= False,
					  width = 1200, height = 1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top Genre')
plt.axis("off")
plt.show()

#Most common genres
Counter([i for j in listOfGenres for i in j]).most_common()

dfTrain['num_genres'] = dfTrain['genres'].apply(lambda x: len(x) if x != {} else 0)
dfTrain['num_genres'].head(10)

dfTrain['all_genres'] = dfTrain['genres'].apply(lambda x:' '.join(sorted([i['name'] for i in x]
																		 if x != {} else '')))
dfTrain['all_genres'].head(10)

topGenres = [m[0] for m in Counter([i for j in listOfGenres for i in j]).most_common(15)]
for g in topGenres:
	dfTrain['genre_' + g] = dfTrain['all_genres'].apply(lambda x: 1 if g in x else 0)

dfTrain = dfTrain.drop(['genres'], axis=1)
 
dfTest['num_genres'] = dfTest['genres'].apply(lambda x: len(x) if x != {} else 0)
dfTest['all_genres'] = dfTest['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x] 
																		if x != {} else '')))

for g in topGenres:
	dfTest['genre_' + g] = dfTest['all_genres'].apply(lambda x: 1 if g in x else 0)

dfTest = dfTest.drop(['genres'], axis=1)

#production Companies

for i, e in enumerate(dfTrain['production_companies'][:5]):
	print(i, e)

print('Number of Production companies in films')
dfTrain['production_companies'].apply(lambda x: len(x) if x != {} else 0).value_counts()

dfTrain[dfTrain['production_companies'].apply(lambda x: len(x) if x != {} else 0) > 11]

#example of the movie poster
from urllib.request import urlopen
img = Image.open(urlopen("https://image.tmdb.org/t/p/w600_and_h900_bestv2/5VKVaTJJsyDeOzY6fLcyTo1RA9g.jpg"))
img 	

listOfComapnies = list(dfTrain['production_companies'].apply(lambda x: [i['name'] for i in x]
															 if x != {} else []).values)

Counter([i for j in listOfComapnies for i in j]).most_common(30)

dfTrain['num_companies'] = dfTrain['production_companies'].apply(lambda x: len(x) if x != {} else 0)
dfTrain['all_production_companies'] = dfTrain['production_companies'].apply(lambda x: ' '.join(sorted(
																			[i['name'] for i in x]
																			if x != {} else '')))

topCompanies = [m[0] for m in Counter(i for j in listOfComapnies for i in j).most_common(30)]
for g in topCompanies:
	dfTrain['production_companies_' + g] = dfTrain['all_production_companies'].apply(lambda x: 1 if g in x else 0)


dfTest['num_companies'] = dfTest['production_companies'].apply(lambda x: len(x) if x != {} else 0)
dfTest['all_production_companies'] = dfTest['production_companies'].apply(lambda x: ' '.join(sorted(
																			[i['name'] for i in x]
																			if x != {} else '')))

for g in topCompanies:
	dfTest['production_companies_' + g] = dfTest['all_production_companies'].apply(lambda x: 1 if g in x else 0)

dfTrain = dfTrain.drop(['production_companies','all_production_companies'], axis = 1)
dfTest = dfTest.drop(['production_companies','all_production_companies'], axis = 1)

#production_countries

for i, e in enumerate(dfTrain['production_countries'][:5]):
	print(i, e)
	
print('Number of production countries in films')
dfTrain['production_countries'].apply(lambda x: len(x) if x != {} else 0).value_counts()

listOfCountries = list(dfTrain['production_countries'].apply(lambda x: [i['name'] for i in x]
															 if x != {} else []).values)

Counter([i for j in listOfCountries for i in j]).most_common(25)

dfTrain['num_countries'] = dfTrain['production_countries'].apply(lambda x: len(x) if x != {} else 0)

dfTrain['all_countries'] = dfTrain['production_countries'].apply(lambda x: ' '.join(sorted(
																	[i['name'] for i in x]
																	if x != {} else '')))
topCountries = [m[0] for m in Counter(i for j in listOfCountries for i in j).most_common(25)]

for g in topCountries:
	dfTrain['production_country_' + g] = dfTrain['all_countries'].apply(lambda x: 1 if g in x else 0)
	

dfTest['num_countries'] = dfTest['production_countries'].apply(lambda x: len(x) if x != {} else 0)
dfTest['all_countries'] = dfTest['production_countries'].apply(lambda x: ' '.join(sorted(
																[i['name'] for i in x]
																if x != {} else '')))

for g in topCountries:
	dfTest['production_countries_' + g] = dfTest['all_countries'].apply(lambda x: 1 if g in x else 0)
	

dfTrain = dfTrain.drop(['production_countries','all_countries'], axis = 1)
dfTest = dfTest.drop(['production_countries','all_countries'], axis = 1)

#spoken language

for i, e in enumerate(dfTrain['spoken_languages'][:5]):
	print(i,e)
	
print('Number of spoken language in the films')

dfTrain['spoken_languages'].apply(lambda x: len(x) if x != {} else 0).value_counts()

listOfLanguages = list(dfTrain['spoken_languages'].apply(lambda x: [i['name'] for i in x]
														 if x != {} else []).values)

Counter([i for j in listOfLanguages for i in j]).most_common(15)

dfTrain['num_languages'] = dfTrain['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

dfTrain['all_languages'] = dfTrain['spoken_languages'].apply(lambda x: ' '.join(sorted(
																[i['name'] for i in x]
																if x != {} else '')))

topLanguages = [m[0] for m in Counter(i for j in listOfLanguages for i in j).most_common(30)]

for g in topLanguages:
	dfTrain['language_' + g] = dfTrain['all_languages'].apply(lambda x: 1 if g in x else 0)

dfTest['num_languages'] = dfTest['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)

dfTest['all_languages'] = dfTest['spoken_languages'].apply(lambda x: ' '.join(sorted(
															[i['name'] for i in x]
															if x != {} else '')))
for g in topLanguages:
	dfTest['language_' + g] = dfTest['all_languages'].apply(lambda x: 1 if g in x else 0)
	
dfTrain = dfTrain.drop(['spoken_languages','all_languages'], axis = 1)
dfTest = dfTest.drop(['spoken_languages','all_languages'], axis = 1)

#Keywords

for i, e in enumerate(dfTrain['Keywords'][:5]):
	print(i, e)
	
print('Number of keywords in films')
dfTrain['Keywords'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)

listOfKeywords = list(dfTrain['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

plt.figure(figsize=(16, 12))
text = ' '.join(['_'.join(i.split(' ')) for j in listOfKeywords for i in j])
wordcloud = WordCloud(max_font_size = None, background_color = 'black', collocations=False, width = 1200,
					  height = 1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top Keywords')
plt.axis("off")
plt.show()

dfTrain['num_keywords'] = dfTrain['Keywords'].apply(lambda x: len(x) if x != {} else 0)
dfTrain['all_keywords'] = dfTrain['Keywords'].apply(lambda x: ' '.join(sorted(
													[i['name'] for i in x]
													if x != {} else '')))
topKeywords = [m[0] for m in Counter([i for j in listOfKeywords for i in j]).most_common(30)]

for g in topKeywords:
	dfTrain['Keywords_' + g] = dfTrain['all_keywords'].apply(lambda x: 1 if g in x else 0)
	
dfTest['num_keywords'] = dfTest['Keywords'].apply(lambda x: len(x) if x != {} else 0)
dfTest['all_keywords'] = dfTest['Keywords'].apply(lambda x: ' '.join(sorted(
													[i['name'] for i in x]
													if x != {} else '')))

for g in topKeywords:
	dfTest['Keywords_' + g] = dfTest['all_keywords'].apply(lambda x: 1 if g in x else 0)
 
dfTrain = dfTrain.drop(['Keywords','all_keywords'], axis = 1)
dfTest = dfTest.drop(['Keywords','all_keywords'], axis = 1)

#cast

for i,e in enumerate(dfTrain['cast'][:1]):
	print(i,e)

print('number of cast memeber in a film')
dfTrain['cast'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)

listOfCastNames = list(dfTrain['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
Counter([i for j in listOfCastNames for i in j]).most_common(15)

listOfCastGender = list(dfTrain['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)
Counter([i for j in listOfCastGender for i in j]).most_common()

listOfCastCharacters = list(dfTrain['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)
Counter([i for j in listOfCastCharacters for i in j]).most_common(15)

dfTrain['num_cast'] = dfTrain['cast'].apply(lambda x: len(x) if x != {} else 0)

topCastName = [m[0] for m in Counter([i for j in listOfCastNames for i in j]).most_common(15)]

for g in topCastName:
	dfTrain['cast_name_' + g] = dfTrain['cast'].apply(lambda x: 1 if g in str(x) else 0)

dfTrain['gender_0_cast'] = dfTrain['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
dfTrain['gender_1_cast'] = dfTrain['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
dfTrain['gender_2_cast'] = dfTrain['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

topCastCharacter = [m[0] for m in Counter([i for j in listOfCastCharacters for i in j]).most_common(15)]

for g in topCastCharacter:
	dfTrain['cast_character_' + g] = dfTrain['cast'].apply(lambda x: 1 if g in str(x) else 0)
	

dfTest['num_cast'] = dfTest['cast'].apply(lambda x: len(x) if x != {} else 0)

for g in topCastName:
	dfTest['cast_name_' + g] = dfTest['cast'].apply(lambda x: 1 if g in str(x) else 0)

dfTest['gender_0_cast'] = dfTest['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
dfTest['gender_1_cast'] = dfTest['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
dfTest['gender_2_cast'] = dfTest['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))


for g in topCastCharacter:
	dfTest['cast_character_' + g] = dfTest['cast'].apply(lambda x: 1 if g in str(x) else 0)

dfTrain = dfTrain.drop(['cast'] , axis = 1)
dfTest = dfTest.drop(['cast'] , axis = 1)
 
#Crew

for i , e in enumerate(dfTrain['crew'][:1]):
	print(i, e[:2])

print('Number of crew in films')
dfTrain['crew'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10)

listOfCrewName = list(dfTrain['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
Counter([i for j in listOfCrewName for i in j]).most_common(15)

listOfCrewJobs = list(dfTrain['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)
Counter([i for j in listOfCrewJobs for i in j]).most_common(15)

listOfCrewGender = list(dfTrain['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)
Counter([i for j in listOfCrewGender for i in j]).most_common(15)

listOfCrewDept = list(dfTrain['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)
Counter([i for j in listOfCrewDept for i in j]).most_common(15)

dfTrain['num_crew'] = dfTrain['crew'].apply(lambda x: len(x) if x != {} else 0)

topCrewNames = [m[0] for m in Counter([i for j in listOfCrewName for i in j]).most_common(15)]

for g in topCrewNames:
	dfTrain['crew_name_' + g] = dfTrain['crew'].apply(lambda x: 1 if g in str(x) else 0)

dfTrain['genders_0_crew'] = dfTrain['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
dfTrain['genders_1_crew'] = dfTrain['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
dfTrain['genders_2_crew'] = dfTrain['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))

topCrewJobs = [m[0] for m in Counter([i for j in listOfCrewJobs for i in j]).most_common(15)]

for j in topCrewJobs:
	dfTrain['jobs_' + j] = dfTrain['crew'].apply(lambda x: sum(1 for i in x if i['job'] == j))
	
topCrewDept = [m[0] for m in Counter([i for j in listOfCrewDept for i in j]).most_common(15)]

for j in topCrewDept:
	dfTrain['departments_' + j] = dfTrain['crew'].apply(lambda x: sum(1 for i in x if i['department'] == j))

dfTest['num_crew'] = dfTest['crew'].apply(lambda x: len(x) if x != {} else 0)

for g in topCrewNames:
	dfTest['crew_name_' + g] = dfTest['crew'].apply(lambda x: 1 if g in str(x) else 0)

dfTest['genders_0_crew'] = dfTest['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
dfTest['genders_1_crew'] = dfTest['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
dfTest['genders_2_crew'] = dfTest['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))


for j in topCrewJobs:
	dfTest['jobs_' + j] = dfTest['crew'].apply(lambda x: sum(1 for i in x if i['job'] == j))
	
for j in topCrewDept:
	dfTest['departments_' + j] = dfTest['crew'].apply(lambda x: sum(1 for i in x if i['department'] == j))

dfTrain = dfTrain.drop(['crew'], axis = 1)
dfTest = dfTest.drop(['crew'], axis = 1)

#Data Exploration

dfTrain.head()

#target variable 
fig, ax = plt.subplots(figsize = (16,6))
plt.subplot(1,2,1)
plt.hist(dfTrain['revenue'])
plt.title('Distribution of revenue')
plt.subplot(1,2,2)
plt.hist(np.log1p(dfTrain['revenue']))
plt.title('Distribution of log of revenue')
plt.show()

dfTrain['log_revenue'] = np.log1p(dfTrain['revenue'])
# exploration of budget
fig, ax = plt.subplots(figsize = (16,6))
plt.subplot(1,2,1)
plt.hist(dfTrain['budget'])
plt.title('Distribution of Budget')
plt.subplot(1,2,2)
plt.hist(np.log1p(dfTrain['budget']))
plt.title('Distribution of log of Budget')
plt.show()

# relationship between budget and revenue
fig, ax = plt.subplots(figsize = (16,8))
plt.subplot(1,2,1)
plt.scatter(dfTrain['budget'], dfTrain['revenue'])
plt.title('Budget vs revenue')
plt.subplot(1,2,2)
plt.scatter(np.log1p(dfTrain['budget']), dfTrain['log_revenue'])
plt.title('log budget vs log revenue')
plt.show()

dfTrain['log_budget'] = np.log1p(dfTrain['budget'])
dfTest['log_budget'] = np.log1p(dfTrain['budget'])

dfTrain['homepage'].value_counts()

dfTrain['has_homepage'] = 0
dfTrain.loc[dfTrain['homepage'].isnull() == False, 'has_homepage'] = 1
dfTest['has_homepage'] = 0
dfTest.loc[dfTest['homepage'].isnull() == False, 'has_homepage'] = 1
# has_homepage vs renenue
sns.catplot(x = 'has_homepage', y = 'revenue', data = dfTrain)
plt.title('Revenue of film with and without homepage')
plt.show() 
# films with a homepage gets more revenue than ones without a homepage. people like more information 

dfTrain['original_language'].value_counts().head(10).index

fig, ax = plt.subplots(figsize = (16,8))
plt.subplot(1,2,1)
sns.boxplot(x = 'original_language', y ='revenue', data = dfTrain.loc[dfTrain['original_language'].isin(
			dfTrain['original_language'].value_counts().head(10).index)])
plt.title('Mean revenue per language')
plt.subplot(1,2,2)
sns.boxplot(x = 'original_language', y ='log_revenue', data = dfTrain.loc[dfTrain['original_language'].isin(
			dfTrain['original_language'].value_counts().head(10).index)])
plt.title('Mean log revenue per language')
plt.show()

plt.figure(figsize = (12,12))
text = ' '.join(dfTrain['original_title'].values)
wordcloud = WordCloud(max_font_size = None, background_color='white', width = 1200, height = 1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top word in title')
plt.axis('off')
plt.show()

plt.figure(figsize = (12,12))
text = ' '.join(dfTrain['overview'].fillna('').values)
wordcloud = WordCloud(max_font_size = None, background_color = 'white', width = 1200, height = 1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in overview')
plt.axis('off')
plt.show()

dfTrain.to_csv(r'train_transformed.csv', index=False, header=True)
dfTest.to_csv(r'test_transformed.csv', index=False, header=True)