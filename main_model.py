import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
import statsmodels.api as sm
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression

city_to_country = {
        'dubai': 'uae',
        'berlin': 'germany',
        'jerusalem': 'israel',
        'haifa': 'israel',
        'beer-sheva': 'israel',
        'natanya': 'israel',
        'rishon': 'israel',
        'tel-aviv': 'israel',
        'ramat-gan': 'israel',
        'nyc': 'usa',
        'boston': 'usa',
        'vegas': 'usa',
        'texas': 'usa',
        'copenhagen': 'denmark',
        'paris': 'france',
        'rome': 'italy',
        'milano': 'italy',
        }

country_to_currency = {
        'israel': 'NIS',
        'uae': 'AED',
        'italy': 'EUR',
        'germany': 'EUR',
        'usa': 'USD',
        'denmark': 'DKK',
        'france': 'EUR',
        }

rates = {
        ('AED', 'NIS'): 0.89,
        ('NIS', 'EAD'): 1.13,
        ('EUR', 'NIS'): 3.99,
        ('NIS', 'EUR'): 0.25,
        ('DKK', 'NIS'): 0.52,
        ('NIS', 'DKK'): 1.91,
        ('USD', 'NIS'): 3.26,
        ('NIS', 'USD'): 0.31,
    }

def get_rate(fromc, toc, date):
    if fromc==toc:
        return 1
    return rates[fromc, toc]

def transform_row(r):
    if len(r.date) == 6:
        r.date += '2021.'
    d = r.date[:-1].split('.')
    r.date = datetime.date(*map(int, d[::-1]))
    r.country = city_to_country[r.city]
    r.currency = country_to_currency[r.country]
    if np.isnan(r.nis):
        r.nis = r.lcy * get_rate(r.currency, 'NIS', r.date)
    r.eur = r.nis * get_rate('NIS', 'EUR', r.date)
    return r

df = pd.read_csv('./expenses.csv')
df = df.apply(transform_row, axis=1)

daily_expenses = []
all_dates = list(pd.date_range(min(df['date']), max(df['date']), freq='D'))
cities = []
for d in list(all_dates):
    value = sum(df[df['date'] == d.date()]['eur'])
    if value:
        cities.append(df[df['date'] == d.date()]['city'].values[-1])
        daily_expenses.append((d.date(), value))
    else:
        all_dates.remove(d)
dates, sums = zip(*daily_expenses)

# defining labeling (one hot encoding)
x = np.array([*zip(range(len(dates)), cities)])
y = sums

preprocess = make_column_transformer((OneHotEncoder(), [-1])).fit_transform(x)
x = np.array([*zip(preprocess, x[:, 0])])

# avoiding the nth label which is redundent
x = x[:, 1:]

# splitting into test set and training set - sampling 0.2 batch
# tts module splits it
xtrain, xtest, ytrain, ytest = tts(x, y, test_size=0.2)

# fitting the regressor to our training set
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# applying the regressor to our test set
ypred = regressor.predict(xtest)

# backward elimination
# removing features that do not have a significant effect on the dependent variable or prediction of output
xopt = np.hstack([np.ones((x.shape[0], 1)), x])
for i in range(xopt.shape[1]):
    pvalues = sm.OLS(y, xopt.astype(np.float64)).fit().pvalues
    mi = np.argmax(pvalues)
    mp = pvalues[mi]
    if mp > 0.05:
        xopt = np.delete(xopt, [mi], 1)
    else:
        break

xtrain, xtest, ytrain, ytest = tts(xopt, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

ypredopt = regressor.predict(xtest)

plt.plot(ytest, color='green')
plt.plot(ypred, color='navy')
plt.plot(ypredopt, color='red')
plt.ylabel('predicted value in eur')
plt.xlabel('days in the test set')
plt.show()

print("linear regression shown")