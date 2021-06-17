import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

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


category_sum = []
for category, rows in df.groupby(['category'])['eur']:
    category_sum.append((sum(rows.values), category))
sums, labels = zip(*sorted(category_sum, reverse=True)[:11])
explode = [0.1]*len(sums)

fig1, ax1 = plt.subplots()
ax1.pie(sums, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')
plt.title('percentage of money spend on each category')
plt.show()