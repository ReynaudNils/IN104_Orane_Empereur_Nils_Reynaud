from IPython.display import display

import pandas as pd
from pdpbox import pdp, get_dataset, info_plots
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

test_titanic = get_dataset.titanic()
print(test_titanic.keys())
titanic_data = test_titanic['data']
titanic_features = test_titanic['features']
titanic_model = test_titanic['rf_model']
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
#PDP for the genderfeature
pdp_sex = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Sex'
)
fig, axes = pdp.pdp_plot(pdp_sex, 'Sex', plot_lines=True, frac_to_plot=0.5)
_ = axes['pdp_ax'].set_xticklabels(['Female', 'Male'])

#Let's go on with the PassengerClass feature
#Firstly, the statistics of survivors based on their PassengerClass
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Pclass', feature_name='Pclass', target=titanic_target, show_percentile=True
)
display(summary_df)
#Prediction of our model on survivors based on their PassengerClass
fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Pclass', feature_name='Pclass', 
    show_percentile=True
)
display(summary_df)
#Let's obtain a PDP on the PassengerClass feature
pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Pclass'
)
fig, axes = pdp.pdp_plot(pdp_fare, 'Pclass',  plot_pts_dist=True)

#Let's roll on with the embark feature
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

#Let's continue with the Age feature
#First let's see the statistics of the survivors depending on their ages
fig, axes, summary_df = info_plots.target_plot(
    df=titanic_data, feature='Age', feature_name='Age', target=titanic_target, show_percentile=True
)
display(summary_df)
#Now let's see the prediction of our model, how many survive for each age category
fig, axes, summary_df = info_plots.actual_plot(
    model=titanic_model, X=titanic_data[titanic_features], feature='Age', feature_name='Age', 
    show_percentile=True
)
display(summary_df)
#And finally let's obtain the PDP for the feature Age
pdp_fare = pdp.pdp_isolate(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, feature='Age'
)
fig, axes = pdp.pdp_plot(pdp_fare, 'Age',  plot_pts_dist=True)

#Let's study the link between Age and Pclass
#Statistics of survivors based on Age and Pclass
fig, axes, summary_df = info_plots.target_plot_interact(
    df=titanic_data, features=['Age', 'Pclass'], feature_names=['Age', 'Pclass'], target=titanic_target
)
display(summary_df.head())
#Prediction of our model, impact if Age and Pclass
fig, axes, summary_df = info_plots.actual_plot_interact(
    model=titanic_model, X=titanic_data[titanic_features], features=['Age', 'Pclass'], feature_names=['Age', 'Pclass']
)
display(summary_df.head())
#PDP for the interaction between Age and Pclass
inter1 = pdp.pdp_interact(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, features=['Age', 'Pclass']
)
fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=inter1, feature_names=['age', 'Pclass'], plot_type='contour', x_quantile=True, plot_pdp=True
)

#Let's study the link between Fare and Sex
#Statistics of survivors based on Fare and Sex
fig, axes, summary_df = info_plots.target_plot_interact(
    df=titanic_data, features=['Fare', 'Sex'], feature_names=['Fare', 'Sex'], target=titanic_target
)
display(summary_df.head())
#Prediction of our model, impact if Fare and Gender
fig, axes, summary_df = info_plots.actual_plot_interact(
    model=titanic_model, X=titanic_data[titanic_features], features=['Fare', 'Sex'], feature_names=['Fare', 'Sex']
)
display(summary_df.head())
#PDP for the interaction between Age and Pclass
inter1 = pdp.pdp_interact(
    model=titanic_model, dataset=titanic_data, model_features=titanic_features, features=['Fare', 'Sex']
)
fig, axes = pdp.pdp_interact_plot(
    pdp_interact_out=inter1, feature_names=['Fare', 'Sex'], plot_type='contour', x_quantile=True, plot_pdp=True
)

plt.show()
