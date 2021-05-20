import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
import xgboost
from sklearn.datasets import make_classification
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
df.drop(columns=['Name'], inplace=True)
test_df.drop(columns=['Name'], inplace=True)
df.drop(columns=['PassengerId'], inplace=True)
test_df.drop(columns=['PassengerId'], inplace=True)


def price_per_passenger(x):
    return round(x.mean() / len(x), 2)

price_per_passenger_df = df.groupby('Ticket').Fare.agg(price_per_passenger).to_frame().reset_index()
price_per_passenger_df.rename(columns=dict(Fare='ticket_price'), inplace=True)
df = df.merge(price_per_passenger_df, on='Ticket', how='left')
df.drop(columns=['Fare','Ticket'], inplace=True)

df.info()
display(df.head())
y_train = df['Survived'].values

# train a model
model = xgboost.train({"learning_rate": 0.01, "seed":100}, 
                      xgboost.DMatrix(df, label=y_train, enable_categorical=True), 
                      100)

# explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df)

shap_values_df = pd.DataFrame(shap_values)
shap_values_df['PassengerId'] = df.index.values
shap_values_df.set_index('PassengerId', inplace=True)

# row_sample = np.random.randint(0, len(X_train_df), 3)
pid_sample = [446, 80, 615, 
             ]


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
for pid in pid_sample:
    passenger = df.loc[[pid]]
    passenger_shap_values = shap_values_df.loc[pid].values
    display(passenger.reset_index().style.hide_index().set_precision(2))#.set_table_styles(styles))
    display(shap.force_plot(explainer.expected_value, passenger_shap_values, passenger))


# visualize the first prediction's explanation 
shap.force_plot(explainer.expected_value, shap_values, df)
shap.summary_plot(shap_values, df)

plt.show()
