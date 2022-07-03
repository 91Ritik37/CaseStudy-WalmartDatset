import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import seaborn as sns
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (15, 10)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

df=pd.read_csv("Walmart.csv")
print(df.info())
'''
  Store         6435 non-null   int64  
 1   Date          6435 non-null   object 
 2   Weekly_Sales  6435 non-null   float64
 3   Holiday_Flag  6435 non-null   int64  
 4   Temperature   6435 non-null   float64
 5   Fuel_Price    6435 non-null   float64
 6   CPI           6435 non-null   float64
 7   Unemployment  6435 non-null   float64
 '''
df.describe()
hist_data = [df.Weekly_Sales]
group_labels = ['Weekly Sales']
sns.scatterplot(x=df.Store, y=df.Weekly_Sales, hue = df.Store);
# plt.show()

corr = df.corr()
f, ax = plt.subplots(figsize=(25,20))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0, annot=True,square=True, linewidths=.5, cbar_kws={'shrink': .5})
# plt.show()



plt.figure(figsize=(15,10))
sns.scatterplot(data=df, x="CPI", y="Weekly_Sales", hue="Weekly_Sales")
plt.show()
def split_date(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week

split_date(df)
df.drop(['Date'], axis=1, inplace=True)
#setting the inputs and the target values
target = df[df.columns[1]]
inputcols =[]
for i in df.columns:
    if i!= "Weekly_Sales":
        inputcols.append(i)
inputs = df[inputcols]
# print(inputs)

from sklearn.model_selection import train_test_split
train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, target, test_size=0.25,  random_state=42)
names = ['Linear Regression', "KNN", "Linear_SVM","Gradient_Boosting", "Decision_Tree", "Random_Forest"]
regressors = [
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=3),
    SVR(kernel="rbf", C=1.0),
    GradientBoostingRegressor(n_estimators=100),
    DecisionTreeRegressor(max_depth=5),
    RandomForestRegressor(max_depth=5, n_estimators=100),
    ]


scores = []
for name, clf in zip(names, regressors):
    clf.fit(train_inputs, train_targets)
    score = clf.score(val_inputs, val_targets)
    scores.append(score)
scores_df = pd.DataFrame()
scores_df['name'] = names
scores_df['score'] = scores
scores_df.sort_values('score', ascending= False)
print(names,scores)

'''
Outputs:
Linear Regression :0.15237105022160158
'KNN'             :0.20101862783127122
Linear_SVM        :-0.034571376460815983
Gradient_Boosting :0.9093821262743614
Decision_Tree     :0.6929379400693525
Random_Forest     :0.7014243104818949
'''