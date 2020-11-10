# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:51:31 2020

@author: sabrine
"""

import pandas as pd
import matplotlib.pyplot as plt




import plotly.express as px

    
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
    r"C:\Users\user\Desktop\projet python_covid19/time_series_covid19_confirmed_global.csv", "Confirmed") \
  .merge(loadData(
    r"C:\Users\user\Desktop\projet python_covid19/time_series_covid19_deaths_global.csv", "Deaths")) \
  .merge(loadData(
    r"C:\Users\user\Desktop\projet python_covid19/time_series_covid19_recovered_global.csv", "Recovered"))

print(allData)
print(allData.describe)
allData.isna().sum()


allData.dtypes.value_counts() #le nombre de type des variables




plt.figure(figsize=(10,8))
plt.bar(allData['date'], allData['Confirmed'],label="Confirmed",color = 'red')
plt.xlabel('date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=8)
plt.title("Confirmation",fontsize=10)
plt.show()

plt.figure(figsize=(10,8))
plt.bar(allData['date'], allData['Recovered'],label="Recovered",color = 'green')
plt.xlabel('date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=8)
plt.title("Recovereed",fontsize=10)
plt.show()


plt.figure(figsize=(10,8))
plt.bar(allData['date'], allData['Deaths'],label="Deaths",color = 'black')
plt.xlabel('date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=8)
plt.title("Deaths",fontsize=10)
plt.show()


plt.figure(figsize=(12,8))
plt.bar(allData['date'], allData['Confirmed'],label="Confirmed")
plt.bar(allData['date'], allData['Recovered'],label="Recovered")
plt.bar(allData['date'], allData['Deaths'],label="Deaths")
plt.xlabel('date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation vs Recoverey vs Death",fontsize=20)
plt.show()



data1=allData[allData['Country/Region'] == 'Tunisia']
print(data1)
plt.figure(figsize=(12,8))
plt.bar(data1['date'], data1['Confirmed'],label="Confirmed",color = 'red')
plt.bar(data1['date'], data1['Recovered'],label="Recovered",color='grey')
plt.bar(data1['date'], data1['Deaths'],label="Deaths",color = 'black')
plt.xlabel('date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation vs Recoverey vs Death in Tunisia ",fontsize=20)
plt.show()


data2=allData[allData['Country/Region'] == 'Spain']
print(data2)
plt.figure(figsize=(12,8))
plt.bar(data2['date'], data2['Confirmed'],label="Confirmed",color = 'blue')
plt.bar(data2['date'], data2['Recovered'],label="Recovered",color = 'yellow')
plt.bar(data2['date'], data2['Deaths'],label="Deaths",color = 'green')
plt.xlabel('date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation vs Recoverey vs Death in Spain ",fontsize=20)
plt.show()



data3=allData[allData['Country/Region'] == 'France']
print(data3)
plt.figure(figsize=(12,8))
plt.bar(data2['date'], data2['Confirmed'],label="Confirmed",color = 'pink')
plt.bar(data2['date'], data2['Recovered'],label="Recovered",color = 'purple')
plt.bar(data2['date'], data2['Deaths'],label="Deaths",color = 'grey')
plt.xlabel('date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation vs Recoverey vs Death in France ",fontsize=20)
plt.show()



data4=allData[allData['Country/Region'] == 'US']
print(data4)
plt.figure(figsize=(12,8))
plt.bar(data2['date'], data2['Confirmed'],label="Confirmed",color = 'orange')
plt.bar(data2['date'], data2['Recovered'],label="Recovered",color = 'yellow')
plt.bar(data2['date'], data2['Deaths'],label="Deaths",color = 'green')
plt.xlabel('date')
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title("Confirmation vs Recoverey vs Death in US ",fontsize=20)
plt.show()

data4=allData[allData['Country/Region'] == 'US']
print((data4['Recovered']).sum())

print((data1['Recovered']).sum())
print((data2['Recovered']).sum())
print((data3['Recovered']).sum())
print((data4['Recovered']).sum())


print((data1['Confirmed']).sum())
print((data2['Confirmed']).sum())
print((data3['Confirmed']).sum())
print((data4['Confirmed']).sum())


print((data1['Deaths']).sum())
print((data2['Deaths']).sum())
print((data3['Deaths']).sum())
print((data4['Deaths']).sum())





allData['Active'] = allData['Confirmed'] -allData['Deaths'] - allData['Recovered']
area_data = allData.groupby(['date'])['Deaths', 'Recovered', 'Active'].sum().reset_index()
area_data = area_data.melt(id_vars="date", value_vars=['Deaths', 'Recovered', 'Active'], var_name='Case', value_name='Count')
fig = px.area(area_data, x="date", y="Count", color='Case',
             title='Whole world Cases over time', color_discrete_sequence = [dth, rec, act])
fig.show()



temp = allData.groupby('date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x="date", y="Count", color='Case', height=600,
             title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()



#plt.figure(figsize=(10,5), dpi = 100)
#sns.lineplot(x = 'date', y = 'Confirmed', data = allData, label = 'Confirmed', color = 'blue', marker = 'o')
#sns.lineplot(x = 'date', y = 'Recovered', data = allData, label = 'Recovered', color = 'green')
#sns.lineplot(x = 'date', y = 'Deaths', data = allData, label = 'Deaths', color = 'black') 
#plt.ylabel('Number of cases')
#plt.legend(loc = 0)
#plt.xticks(rotation = 90)
#plt.tight_layout()
#plt.show()





































#missing data
total = allData.isnull().sum().sort_values(ascending=False)
percent_1 = allData.isnull().sum()/allData.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data)

#############################################################

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








#cercle
group_size = [sum(allData['Confirmed']),sum(allData['Recovered']),sum(allData['Deaths'])]
group_labels = ['Confirmed\n' + str(sum(allData['Confirmed'])),
                'Recovered\n' + str(sum(allData['Recovered'])), 
                'Deaths\n'  + str(sum(allData['Deaths']))]
custom_colors = ['skyblue','yellowgreen','tomato']
plt.figure(figsize = (5,5))
plt.pie(group_size, labels = group_labels, colors = custom_colors)
central_circle = plt.Circle((0,0), 0.5, color = 'white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size = 12) 
plt.title('Nationwide total Confirmed, Recovered and Deceased Cases', fontsize = 16)
plt.show()



















