from IPython.display import display

import pandas as pd
from pdpbox import pdp, get_dataset, info_plots
import xgboost
from xgboost import XGBClassifier
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

test_titanic = get_dataset.titanic()
print(test_titanic.keys())

titanic_data = test_titanic['data']
titanic_features = test_titanic['features']
titanic_model = test_titanic['xgb_model']
titanic_target = test_titanic['target']

#Let's start with the gender
#Survivors based on their sex
fig, axes, summary_df = info_plots.target_plot(
	df=titanic_data, feature='Sex', feature_name='gender', target=titanic_target
	)
_ = axes['bar_ax'].set_xticklabels(['Female', 'Male'])

display(summary_df)

#Chance of survival our model give based on gender
fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Sex', feature_name='gender'
)
display(summary_df)

pdp_sex = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Sex'
)
fig, axes = pdp.pdp_plot(pdp_sex, 'Sex', plot_lines=True, frac_to_plot=0.5)
_ = axes['pdp_ax'].set_xticklabels(['Female', 'Male'])

#Let's continue with the embark feature
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'], feature_name='embarked', 
    target=titanic_target
)
display(summary_df)

fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature=['Embarked_C', 'Embarked_S', 'Embarked_Q'], 
    feature_name='embarked'
)
display(summary_df)

pdp_embark = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, 
    feature=['Embarked_C', 'Embarked_S', 'Embarked_Q']
)
fig, axes = pdp.pdp_plot(pdp_embark, 'Embark')

#Now with the Fare feature
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Fare', feature_name='fare', target=titanic_target, show_percentile=True
)
display(summary_df)
fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Fare', feature_name='Fare', 
    show_percentile=True
)
display(summary_df)
pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Fare'
)
fig, axes = pdp.pdp_plot(pdp_fare, 'Fare',  plot_pts_dist=True)

#Let's study the link between Age and Fare
fig, axes, summary_df = info_plots.target_plot_interact(
    df=titanic_data, features=['Age', 'Fare'], feature_names=['Age', 'Fare'], target=titanic_target
)
display(summary_df.head())


fig, axes, summary_df = info_plots.actual_plot_interact(
    model=titanic_model, X=titanic_data[titanic_features], features=['Age', 'Fare'], feature_names=['Age', 'Fare']
)
display(summary_df.head())


inter1 = pdp.pdp_interact(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, features=['Age', 'Fare']
)
fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=inter1, feature_names=['age', 'fare'], plot_type='contour', x_quantile=True, plot_pdp=True
)

plt.show()
