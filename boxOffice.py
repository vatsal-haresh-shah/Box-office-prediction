# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 10:03:04 2020

@author: Vatsal Shah
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
import eli5
from IPython.display import display, HTML

train = pd.read_csv('train_transformed.csv')
test = pd.read_csv('test_transformed.csv')

vectorizer = TfidfVectorizer(sublinear_tf = True,
							 analyzer = 'word',
							 token_pattern = r'\w{1,}',
							 ngram_range = (1,2),
							 min_df=5)

overview_text = vectorizer.fit_transform(train['overview'].fillna(''))

linreg = LinearRegression()

linreg.fit(overview_text, train['log_revenue'])
e = eli5.show_weights(linreg, vec = vectorizer, top = 20, feature_filter = lambda x: x != '<BIAS>')

display(e)




