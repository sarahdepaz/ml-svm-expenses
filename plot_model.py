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
print("percentage of money spend on each category plot created")

preferred_transport = []
for desc, rows in df.groupby(['description']):
    if all(i in ['travel', 'transport'] for i in rows['category']):
        preferred_transport.append((sum(rows['eur'].values), desc))

sums, labels = zip(*sorted(preferred_transport, reverse=True))
explode = [0.1]*len(sums)

fig1, ax1 = plt.subplots()
ax1.pie(sums, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=0)
ax1.axis('equal')
plt.title('preferred transport')
plt.show()
print("preferred transport plot created")

all_categories = tuple(set(df['category']) - set('travel'))
cities_daily = []
for city, rows in df.groupby(['city']):
    days = set(rows['date'].values)
    days = (max(days) - min(days)).days + 1
    descs = {desc: sum(rs['eur'].values)/days for desc, rs in rows[rows['category'] != 'travel'].groupby(['category'])}
    cities_daily.append((city, tuple(descs[i] if i in descs else 0 for i in all_categories)))

cities, sums = zip(*sorted(cities_daily, reverse=True, key=lambda t: sum(t[1])))
sums = list(zip(*sums))
width = 0.35
ind = np.arange(len(cities))
colors = ['maroon','c','orange','k','b','darkmagenta','g','m','yellow','r','peru','navy','cyan','plum','grey','teal','lime']
bars = [plt.bar(ind, sums[0], width, color=colors[0])]
for i in range(1, len(all_categories)):
    bars.append(plt.bar(ind, sums[i], width, bottom=list(map(sum, zip(*sums[:i]))), color=colors[i]))

plt.title('amount of money spent daily per city')

plt.xticks(np.arange(len(cities)), cities)
plt.yticks(np.arange(0, 26, 3))
plt.tick_params(labelsize=6)
plt.legend(list(zip(*bars))[0], all_categories)
plt.show()
print("amount of money spent daily per city plot")

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

ind = np.arange(len(all_dates))
plt.bar(ind, sums, color='red', width=0.35)
plt.xticks(ind, list(range(len(all_dates))))
plt.title('daily amount of money spend')
plt.xlabel('day number')
plt.ylabel('amount of money in eur')
plt.show()

print("amount of money in eur plot")