from IPython.display import display

import pandas as pd
from pdpbox import pdp, get_dataset, info_plots
import xgboost
from xgboost import XGBClassifier
print(xgboost.__version__)
import matplotlib
print(matplotlib.__version__)
import matplotlib.pyplot as plt
import sklearn
print(sklearn.__version__)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('titanic/train.csv')
print(train_df.shape)
display(train_df.head())

titanic_data = train_df.copy()
titanic_data = titanic_data.drop('Cabin', axis=1)
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())
titanic_data['Embarked']= titanic_data['Embarked'].fillna(titanic_data['Embarked'].value_counts().index[0])
LE = LabelEncoder()
titanic_data['Sex'] = LE.fit_transform(titanic_data['Sex'])
for v in ['C', 'S', 'Q']:
    titanic_data['Embarked_{}'.format(v)] = (titanic_data['Embarked'] == v).map(int)

features= ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_S', 'Embarked_Q']
target = ['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    titanic_data[features], titanic_data[target], test_size = 0.2, random_state=42)

classifier = XGBClassifier(
    max_depth=12,
    subsample=0.33,
    objective='binary:logistic',
    n_estimators=100,
    learning_rate = 0.01)
eval_set = [(X_train,y_train), (X_test,y_test)]

classifier.fit(
    X_train, y_train.values.ravel(), 
    early_stopping_rounds=12, 
    eval_metric=["error", "logloss"],
    eval_set=eval_set, 
    verbose=True
)
classifier.score(X_test,y_test)

titanic_features = features
titanic_model = classifier
titanic_target = 'Survived'

test_titanic = get_dataset.titanic()
print(test_titanic.keys())

titanic_data = test_titanic['data']
titanic_features = test_titanic['features']
titanic_model = test_titanic['xgb_model']
titanic_target = test_titanic['target']

fig, axes, summary_df = info_plots.target_plot(
	df=titanic_data, feature='Sex', feature_name='gender', target=titanic_target
	)
_ = axes['bar_ax'].set_xticklabels(['Female', 'Male'])

display(summary_df)

pdp_sex = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Sex'
)
fig, axes = pdp.pdp_plot(pdp_sex, 'Sex')
_ = axes['pdp_ax'].set_xticklabels(['Female', 'Male'])
plt.show()
