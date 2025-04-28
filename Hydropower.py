import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nepali_datetime
import datetime
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score,train_test_split



#importing data
file = r"C:\Users\DELL\Downloads\764807bc-1927-4576-acc7-c5b4bc4afd5b.csv"
df = pd.read_csv(file)


#EDA
df.head()
df.describe()
df.info()

#cleaning data
def clean(bs_date_time):
    if isinstance(bs_date_time, nepali_datetime.date):
        return bs_date_time

    if pd.isnull(bs_date_time) or str(bs_date_time).strip() == '':
        return None

    bs_date_str = str(bs_date_time)

    if 'T' in bs_date_str:
        bs_date_part = bs_date_str.split('T')[0]
    else:
        bs_date_part = bs_date_str

    bs_date_part = bs_date_part.replace('.', '-')

    parts = bs_date_part.split('-')
    if len(parts) != 3:
        return None  

    try:
        year, month, day = map(int, parts)
        return nepali_datetime.date(year, month, day)
    except ValueError:
        return None  

df['PPA Date'] = df['PPA Date'].apply(clean)
df['Commercial Operation Date'] = df['Commercial Operation Date'].apply(clean)
    
print(df.to_string())


#data was conberted from ad to bs in externally
df.replace({'Commercial Operation Date':{None:'2062-09-17'}},inplace = True)
df.replace({'Location':{'Sindhupalchok':'Sindhupalchowk'}},inplace = True)
df.replace({'Location':{'Panchtar':'Panchthar'}},inplace = True)
df.replace({'Location':{'Ilam':'Illam'}},inplace = True)
df.replace({'Location':{'Dolkha':'Dolakha'}},inplace = True)


df.head(10)
df.dtypes

#Feature engineering
x = df['PPA Date'].apply(lambda y: nepali_datetime.date(y.year, y.month, y.day))
today = nepali_datetime.date.today()
df["Project Age"] = (today - x).apply(lambda z: z.days / 365)


df['Project Age'] = np.round(df['Project Age'],0)


df['Capacity (kW)'] = df['Capacity (kW)']/1000
df.rename(columns = {'Capacity (kW)':"Capacity_MW"},inplace = True)

o = df['Commercial Operation Date'].apply(lambda u:nepali_datetime.date(u.year,u.month,u.day))
df['Completion time'] = (o - x).apply(lambda g:g.days/365)
df['Completion time'] = np.round(df['Completion time'],2)

df.head()


df = df[df['Completion time'] != -10.54]
#Descriptive Statistics
Avg_capacity_MW = df['Capacity_MW'].mean()
print(f'The average capacity is {Avg_capacity_MW:.2f} MW')
Avg_project_age = df['Project Age'].mean()
print(f'The average project age is {Avg_project_age:.2f} years')
Avg_completion_time = df['Completion time'].mean()
print(f'The average completion time is {Avg_capacity_MW:.2f} years \n')

min_index = df['Capacity_MW'].idxmin()
print('Project with min capacity description :\n',df.loc[min_index])

max_index = df['Capacity_MW'].idxmax()
print('\nProject with min capacity description :\n',df.loc[max_index])



#Capacity  vs Completion time
x = df['Capacity_MW']
y = df['Completion time']
plt.figure(figsize = (10,6))
plt.title("Capacity vs Completion Time")
plt.xlabel("Capacity (MW)")
plt.ylabel("Completion Time (yrs)")
plt.scatter(x,y,edgecolors = 'r',linewidth = 0.75)
plt.show()



#PPA Date vs Completion time
x = df['PPA Date'].apply(lambda y: nepali_datetime.date(y.year, y.month, y.day).year).astype(int)
y = df['Completion time']
plt.figure(figsize = (10,6))
plt.title("PPA Date vs Completion Time")
plt.xlabel("PPA Date (yrs)")
plt.ylabel("Completion Time (yrs)")
plt.scatter(x,y,edgecolors = 'r',linewidth = 0.75)
plt.show()


#PPA Date vs no of projects
df['ppa_year']= df['PPA Date'].apply(lambda y: nepali_datetime.date(y.year, y.month, y.day).year).astype(int)
grouping = df.groupby(['ppa_year']).size().reset_index(name = 'no of project')

plt.figure(figsize = (10,6))
plt.title("PPA Date vs No of Projects")
plt.xlabel('PPA year')
plt.ylabel('no of project signed')
plt.bar(grouping['ppa_year'],grouping['no of project'])
plt.xticks(rotation = 90)
plt.show()



#Location vs total capacity
df_grp_loc = df[['Location','Capacity_MW']]
df_grouped = df_grp_loc.groupby(['Location'],as_index = False).agg({'Capacity_MW':'sum'})


x = df_grouped['Location']
y = df_grouped['Capacity_MW']
plt.figure(figsize = (10,6))
plt.title('Location vs Capacity')
plt.xlabel('Location')
plt.ylabel('Capacity (MW)')
plt.barh(x,y)
plt.show()


#Location vs Completion Time
df_grp_loc1 = df[['Location','Completion time']]
df_grouped1 = df_grp_loc1.groupby(['Location'],as_index = False).agg({'Completion time':'mean'})
x = df_grouped1['Location']
y = df_grouped1['Completion time']
plt.figure(figsize = (10,6))
plt.title('Location vs Completion Time')
plt.xlabel('Location')
plt.ylabel('Completion Time')
plt.barh(x,y)
plt.show()



df.head()




df_grp_loc2 = df[['ppa_year','Completion time']]
df_grouped2 = df_grp_loc2.groupby(['ppa_year'],as_index = False).agg({'Completion time':'mean'})
x = df_grouped2['ppa_year']
y = df_grouped2['Completion time']
plt.figure(figsize = (10,6))
plt.title('Completion Time Over Years')
plt.xlabel("Year")
plt.ylabel("Completion Time")
plt.plot(x,y, marker = 'o',ls = ':')



#correlation
df[['Completion time','Capacity_MW']].corr()




pearson_coef,p_value = stats.pearsonr(df['Completion time'],df['Capacity_MW'])
print('The Pearson Correlation Coefficient is',pearson_coef,'with a P-value of P =',p_value)
sns.regplot(x = 'Completion time', y = 'Capacity_MW',data = df,line_kws = {'color': 'red'}) 



df[['Completion time','Project Age']].corr()



pearson_coef,p_value = stats.pearsonr(df['Completion time'],df['Project Age'])
print('The Pearson Correlation Coefficient is',pearson_coef,'with a P-value of P =',p_value)
sns.regplot(x = 'Completion time', y = 'Project Age',data = df,line_kws = {'color': 'red'}) 




df[['Capacity_MW','Project Age']].corr()
pearson_coef,p_value = stats.pearsonr(df['Capacity_MW'],df['Project Age'])
print('The Pearson Correlation Coefficient is',pearson_coef,'with a P-value of P =',p_value)
sns.regplot(x = 'Capacity_MW', y = 'Project Age',data = df,line_kws = {'color': 'red'}) 




df_gptest = df[['Capacity_MW','Completion time','Project Age']]
grouped_test1 = df_gptest.groupby(['Capacity_MW','Completion time'],as_index=False).mean()



grouped_pivot = grouped_test1.pivot(index='Capacity_MW',columns='Completion time')
grouped_pivot = grouped_pivot.fillna(0)



fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

plt.xticks(rotation=90)
plt.figure(figsize = (200,100))
fig.colorbar(im)
plt.show()


#Regression Model
##Between Capacity and Completion Time
lm = LinearRegression()
lm
x = df[['Capacity_MW']]
y = df['Completion time']
lm.fit(x,y)
yhat = lm.predict(x)
yhat[0:5]

lm.intercept_
lm.coef_


print('The R-square is:',lm.score(x,y))


mse = mean_squared_error(df['Completion time'], yhat)
print("The mean squared error of completion Time and predicted value is:",mse)


plt.figure(figsize=(8, 6))

# Scatter plot: Actual vs Predicted
plt.scatter(y, yhat, color='blue', edgecolors='k', alpha=0.7)

# Plot a diagonal reference line (perfect prediction line)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)

plt.xlabel('Actual Completion Time')
plt.ylabel('Predicted Completion Time')
plt.title('Actual vs Predicted Completion Time')
plt.grid(True)
plt.show()



#Multiple Regression
lm2 = LinearRegression()
z= df[['Capacity_MW','Project Age']]
lm2.fit(z,y)
lm2.coef_


lm2.intercept_



yhat = lm2.predict(z)


ax1 = sns.kdeplot(df['Completion time'], color="r", label="Actual Value")
sns.kdeplot(yhat, color="b", label="Fitted Values", ax=ax1)

plt.title('Actual vs Fitted values for Completion time')
plt.xlabel('Completion time')
plt.ylabel('Density')

plt.legend()
plt.show()
plt.close()


lm.fit(z, y)
print('The R-square is: ', lm.score(z, y))


print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(y,yhat))


df.to_csv('Cleaned_Hydropower_projects_nepal.csv',index = False)
