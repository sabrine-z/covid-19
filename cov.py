# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:38:05 2020

@author: user
"""

""
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
# color pallette
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 
print(allData.shape)#taille de fichier
allData.head()  #affichier les premiers lignes
print(allData.describe())# utilisée pour calculer certaines données statistiques comme le count mean.....
allData.dtypes#le nombre de type des variablesdata.columns#afficher les noms des colonnes de ce fichier
allData.info()

def loadData(fileName, columnName):
    data = pd.read_csv(fileName) \
             .melt(id_vars=['Province/State', 'Country/Region','Lat','Long'], 
                 var_name='date', value_name=columnName) \
             .astype({'date':'datetime64[ns]', columnName:'Int64'}, 
                 errors='ignore')
    data['Province/State'].fillna('<all>', inplace=True)
    data[columnName].fillna(0, inplace=True)
    return data
allData = loadData(
    "C:/Users/islem/Desktop/S2/PYTHON/projet python/time_series_covid19_confirmed_global.csv", "CumConfirmed") \
  .merge(loadData(
    "C:/Users/islem/Desktop/S2/PYTHON/projet python/time_series_covid19_deaths_global.csv", "CumDeaths")) \
  .merge(loadData(
    "C:/Users/islem/Desktop/S2/PYTHON/projet python/time_series_covid19_recovered_global.csv", "CumRecovered"))

print(allData.to_string())
allData.columns

allData['CumConfirmed'].plot(figsize=(9,6))# C BON111111111




#cbon    somme de confirmed selon une date 
cc_by_country = allData[allData['date'] == '2020-03-18'].groupby('Country/Region').sum()['CumConfirmed'].reset_index()
fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, batches = ax.hist(cc_by_country['CumConfirmed'])
#
group_size = [sum(allData['CumConfirmed']),#cbon
              sum(allData['CumRecovered']),
              sum(allData['CumDeaths'])]
group_labels = ['CumConfirmed\n'+ str(sum(allData['CumConfirmed'])),
                'CumRecivered\n' + str(sum(allData['CumRecovered'])),
                'CumDeaths\n' + str(sum(allData['CumDeaths']))]
custom_colors = ['skyblue','yellowgreen','tomato']
plt.figure(figsize = (5,5))
plt.pie(group_size, labels = group_labels, colors = custom_colors)
central_circle = plt.Circle((0,0), 0.5, color = 'white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
#mapp
import folium
worldmap = folium.Map(location=[32.4279,53.6880 ], zoom_start=4,tiles='Stamen Toner')

for Lat, Long, confirm,death,recover in zip(allData['Lat'], allData['Long'],allData['Confirmed'],allData['Deaths'],allData['Recovered']):
    folium.CircleMarker([Lat, Long],
                        radius=5,
                        color='red',
                      popup =('Confirmed: ' + str(confirm) + '<br>'
                              'Recovered: ' + str(recover) + '<br>'
                              'Deaths: ' + str(death) + '<br>'
                             ),
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(worldmap)





#


train_df_jp = allData[allData['Country/Region'] == 'Thailand']

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.bar(train_df_jp['date'], train_df_jp['CumConfirmed'])

#c bon
import statsmodels.graphics.api as smg
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(train_df_jp['CumConfirmed'], lags=29, alpha=None)
plt.show()

#
#======confirmed================



allData[['CumConfirmed','CumDeaths']].plot(figsize=(9,10), linewidth=3, fontsize=10)
plt.xlabel('date', fontsize=10);
plt.show()

#=======




allData['CcDiff'] = allData['CumConfirmed'].diff().fillna(0)

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
ax.bar(allData['date'], allData['CcDiff'])

#########----------------------------------------------------------------------
train_df_jp = allData[allData['Country/Region'] == 'Japan']

fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111)
ax.bar(train_df_jp['date'], train_df_jp['CumConfirmed'])
#-------------------

train_df_ch=allData[allData['Country/Region'] == 'Turkey'].groupby('date').sum()['CumConfirmed'].reset_index()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(train_df_ch['date'], train_df_ch['CumConfirmed'])

liste=['Italy','Tunisia','France','China']
for i in liste:
    countries_data=countries_datas.append(allData[allData['Country/Region'] == i])
sns.set_style("ticks")
plt.figure(figsize = (9,5))
plt.barh(countrydatas["Country/Region"],countrydatas["CumConfirmed"].map(int),
         align = 'center', color = 'lightblue', edgecolor = 'blue')
plt.xlabel('No. of Confirmed cases', fontsize = 18)
plt.ylabel('Country/Region', fontsize = 18)
plt.gca().invert_yaxis() # this is to maintain the order in which the states appear
plt.xticks(fontsize = 14) 
plt.yticks(fontsize = 14)
plt.title('Total Confirmed Cases Statewise', fontsize = 20)

for index, value in enumerate(allData["CumConfirmed"]):
    plt.text(value, index, str(value), fontsize = 12, verticalalignment = 'center')
plt.show()  



# barplot to show total confirmed cases Statewise 

states = allData['Province/State'].unique()
countries = allData['Country/Region'].unique()
confirm_dict = {}
deaths_dict = {}
recover_dict = {}
for country in countries:
    country_data = allData[allData['Country/Region'] == country]
    max_date = country_data['date'].max()
    sub = country_data[country_data['date'] == max_date]
    confirm = sub['Confirmed'].sum()
    death = sub['Deaths'].sum()
    recover = sub['Recovered'].sum()
    
    confirm_dict[country] = confirm
    deaths_dict[country] = death
    recover_dict[country] = recover
confirm_dict_sorted = sorted(confirm_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
deaths_dict_sorted = sorted(deaths_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
recover_dict_sorted = sorted(recover_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
confirm_dict_sorted = sorted(confirm_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
deaths_dict_sorted = sorted(deaths_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
recover_dict_sorted = sorted(recover_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
plt.figure(figsize = (7,6))
top10_confirm = confirm_dict_sorted[:10]
top10_deaths = deaths_dict_sorted[:10]
top10_recover = recover_dict_sorted[:10]
top10_confirm = dict(top10_confirm)
top10_deaths = dict(top10_deaths)
top10_recover = dict(top10_recover)
bars = plt.bar(top10_confirm.keys(), top10_confirm.values())
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Highest Confirmed Cases in 10 countries')
plt.xticks(list(top10_confirm.keys()), rotation = 90)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval, rotation = 5)
plt.show()

plt.figure(figsize = (7,6))
bars = plt.bar(top10_deaths.keys(), top10_deaths.values())
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Highest Death Cases in 10 countries')
plt.xticks(list(top10_deaths.keys()), rotation = 90)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval, rotation = 5)
plt.show()

plt.figure(figsize = (7,6))
bars = plt.bar(top10_recover.keys(), top10_recover.values())
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Highest Recovered Cases in 10 countries')
plt.xticks(list(top10_recover.keys()), rotation = 90)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x(), yval + .005, yval, rotation = 5)
plt.show()




def plot_count(feature, value, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    df = df.sort_values([value], ascending=False).reset_index(drop=False)
    g = sns.barplot(df[feature][0:10], df[value][0:10], palette='Set3')
    g.set_title("Number of {} - first 10 by number".format(title))
    ax.set_xticklabels(ax.get_xticklabels(),rotation=50)
    plt.show() 


data_ct = allData.sort_values(by = ['Country/Region','date'], ascending=False)
filtered_data_ct_last = allData.drop_duplicates(subset = ['Country/Region'], keep='first')
data_ct_agg = data_ct.groupby(['date']).sum().reset_index()
plot_count('Country/Region', 'Confirmed', 'Confirmed cases - all World', filtered_data_ct_last, size=4)
 


#confirmed selon la date
allData['Confirmed'].plot(figsize=(9,6))
cc_by_country =allData[allData['Date'] == '2020-03-18'].groupby('Country/Region').sum()['Confirmed'].reset_index()
fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, batches = ax.hist(cc_by_country['Confirmed'])

allData['Confirmed'].plot(figsize=(9,6))
cc_by_country =allData[allData['Date'] == '2020-01-18'].groupby('Country/Region').sum()['Confirmed'].reset_index()
fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, batches = ax.hist(cc_by_country['Confirmed'])





allData['Deaths'].plot(figsize=(9,6))
cc_by_country =allData[allData['Date'] == '2020-03-18'].groupby('Country/Region').sum()['Deaths'].reset_index()
fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, batches = ax.hist(cc_by_country['Deaths'])


allData['Recovered'].plot(figsize=(9,6))
cc_by_country =allData[allData['Date'] == '2020-01-18'].groupby('Country/Region').sum()['Recovered'].reset_index()
fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, batches = ax.hist(cc_by_country['Recovered'])




allData['Recovered'].plot(figsize=(9,6))
cc_by_country =allData[allData['Date'] == '2020-03-18'].groupby('Country/Region').sum()['Recovered'].reset_index()
fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, batches = ax.hist(cc_by_country['Recovered'])






train_df_tn =allData[allData['Country/Region'] == 'Tunisia'].groupby('date').sum()['Confirmed'].reset_index()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(train_df_tn['date'], train_df_tn['Confirmed'])




train_df_tn =allData[allData['Country/Region'] == 'France'].groupby('date').sum()['Confirmed'].reset_index()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(train_df_tn['date'], train_df_tn['Confirmed'])


train_df_tn =allData[allData['Country/Region'] == 'France'].groupby('date').sum()['Confirmed'].reset_index()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(train_df_tn['date'], train_df_tn['Recovered'])



train_df_tn =allData[allData['Country/Region'] == 'US'].groupby('date').sum()['Confirmed'].reset_index()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(train_df_tn['date'], train_df_tn['Confirmed'])



train_df_tn =allData[allData['Country/Region'] == 'Spain'].groupby('date').sum()['Confirmed'].reset_index()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(train_df_tn['date'], train_df_tn['Confirmed'])




train_df_tn =allData[allData['Country/Region'] == 'Tunisia'].groupby('date').sum()['Confirmed'].reset_index()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(train_df_tn['date'], train_df_tn['Recovered'])




train_df_tn =allData[allData['Country/Region'] == 'Tunisia'].groupby('date').sum()['Confirmed'].reset_index()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(train_df_tn['date'], train_df_tn['Deaths'])

#-----


train_df_tn =allData[allData['Country/Region'] == 'Italie'].groupby('date').sum()['Confirmed'].reset_index()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(train_df_tn['date'], train_df_tn['Confirmed'])



#-------------------

train_df_ch=allData[allData['Country/Region'] == 'Turkey'].groupby('date').sum()['Confirmed'].reset_index()
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
ax.bar(train_df_ch['date'], train_df_ch['Confirmed'])




























def plot_time_variation_all(df, title='Mainland China', size=1):
    f, ax = plt.subplots(111,111, figsize=(4*size,2*size))
    g = sns.lineplot(x="date", y='Confirmed', data=df, color='blue', label='Confirmed')
    g = sns.lineplot(x="date", y='Recovered', data=df, color='green', label='Recovered')
    g = sns.lineplot(x="date", y='Deaths', data=df, color = 'red', label = 'Deaths')
    plt.xlabel('date')
    plt.ylabel(f'Total {title} cases')
    plt.xticks(rotation=90)
    plt.title(f'Total {title} cases')
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show() 

plot_time_variation_all(allData, 'All World', size=3)







allData.plot(figsize = (9,6)x,x='CumConfirmed',y='date', marker = "*", color = 'teal',linestyle = 'dashed', linewidth =1)
plt.title("Total confirmed case of covid19")
plt.xticks(rotation=90)
plt.show()

t = allData.groupby(['Country/Region','Province/State']).agg({'CumConfirmed':'max'})
t = t.loc[t['CumConfirmed'] > 50]
Data = pd.merge(allData,t[[]],left_on=['Country/Region','Province/State'], right_index=True)
Data['Country/Region'].value_counts()

result = allData.groupby('Country/Region').max().sort_values(by='CumConfirmed', ascending=False)[:10]
pd.set_option('display.max_column', None)
print(result)


#Write a Python program to list countries with no cases of Novel Coronavirus (COVID-19) recovered.
import pandas as pd
data = allData.groupby('Country/Region')['CumConfirmed', 'CumDeaths', 'CumRecovered'].sum().reset_index()
result = data[data['CumRecovered']==0][['Country/Region', 'CumConfirmed', 'CumDeaths', 'CumRecovered']]
print(result)

#Write a Python program to get the latest country wise deaths cases of Novel Coronavirus (COVID-19).
data = allData.groupby('Country/Region')['CumConfirmed', 'CumDeaths', 'CumRecovered'].sum().reset_index()
result = data[data['CumDeaths']>0][['Country/Region', 'CumDeaths']]
print(result)


data = allData.groupby('Country/Region')['CumConfirmed', 'CumDeaths', 'CumRecovered'].max()
pd.set_option('display.max_rows', None)
print(data)
print(allData)
print("\nDataset information:")
print(allData.info())
print("\nMissing data information:")
print(allData.isna().sum())

import numpy as np
# Highlight a column in a table
def highlight_cols(s, coldict):
    if s.name in coldict.keys():
            return ['background-color: {}'.format(coldict[s.name])]*len(s)
    return ['']*len(s)
# Adding a text label in a bar chart
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
# Getting the sum for each country and sorting it by a number of cases
result = allData.groupby(['Country/Region']).sum()
result = result.sort_values(['CumConfirmed'], ascending=False)
# Limiting to top 10
labels = result.index.values[:10]
confirmed = result['CumConfirmed'][:10]
# Chart configuration
width = 0.35
fig, ax = plt.subplots()
fig.autofmt_xdate()
ax.bar(labels, confirmed, width, label='Confirmed cases', color='blue')
ax.set_ylabel('Number of cases')
ax.set_title('Top 10 most cases by country')
ax.legend()
x = np.arange(len(labels))
rects1 = ax.bar(x, confirmed, width, color='blue')
autolabel(rects1)
# Chart display
plt.figure(figsize = (20,26))

# Showing the results in a table and highlighting sorted by column
coldict = {'CumConfirmed': 'rgb(230,220,240)'}
result[['CumConfirmed', 'CumDeaths', 'CumRecovered']].head(10).style.apply(highlight_cols, coldict=coldict)
