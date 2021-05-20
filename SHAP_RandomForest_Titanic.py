import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from IPython.core.display import HTML
from IPython.display import display
import shap

df = pd.read_csv('titanic/train.csv')
display(df.head())
test_df = pd.read_csv("titanic/test.csv")


#See the summary of data and see the missing values in Age and Cabin
df.info()
df['Age'].fillna(df['Age'].median(), inplace=True)

df.drop(columns=['Cabin'], inplace=True)
test_df.drop(columns=['Cabin'], inplace=True)

genders = {"male": 0, "female": 1}
embarkat = {"S": 0, "C": 1, "Q": 1}
data = [df, test_df]
for dataset in data:
    
    #convert gender to a number
    dataset['is_female'] = dataset['Sex'].map(genders)
    #convert embarked to a number
    dataset['Embarked_'] = dataset['Embarked'].map(embarkat)

df.drop(columns=['Sex'], inplace=True)
test_df.drop(columns=['Sex'], inplace=True)
df.drop(columns=['Embarked'], inplace=True)
test_df.drop(columns=['Embarked'], inplace=True)


def price_per_passenger(x):
    return round(x.mean() / len(x), 2)

price_per_passenger_df = df.groupby('Ticket').Fare.agg(price_per_passenger).to_frame().reset_index()
price_per_passenger_df.rename(columns=dict(Fare='ticket_price'), inplace=True)
df = df.merge(price_per_passenger_df, on='Ticket', how='left')
df.drop(columns=['Fare','Ticket'], inplace=True)

df.info()
display(df.head())

df['PassengerId'] = df.fillna(df.mean())
display(df.head())
X = df[['PassengerId','Pclass','Age','SibSp','Parch','ticket_price','is_female']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, y_train)

#generate predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)**(0.5)
print(mse)

# explain the model's predictions with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# visualize the first prediction's explanation 
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_train)
shap_values = shap.TreeExplainer(model).shap_values(X_train)
#Let's class the features by their importance
shap.summary_plot(shap_values, X_train, plot_type="bar")
#SHAP Summary plot
shap.summary_plot(shap_values, X_train)
#SHAP Dependance plot
shap.dependence_plot('Age', shap_values, X_train)
plt.show()
