---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Hypothesis

So far we have gathered enough information to make some hypothesis

## Economic shock

+ We can see a negative impact on renewals during 2 economic shocks: subscriptions were affected by the economic conditions around 2021 where inflation and gdp growth were very distant and had a negative correlation: almost suggesting a a slowdown or recession.

+ Just by looking at both graphs before we will delimitate the shocks to occur from 2019-Q2 to 2020-Q2 and from 2021-Q3 to 2022-Q4.


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

# Re-run this cell
# Import required libraries
import pandas as pd
from src.utils import *
# Import data
client_details = pd.read_csv('data/client_details.csv')
subscription_records = pd.read_csv('data/subscription_records.csv', parse_dates = ['start_date','end_date'])
economic_indicators = pd.read_csv('data/economic_indicators.csv', parse_dates = ['start_date','end_date'])
df = client_details.merge(subscription_records)
df['end_date'] = df['end_date'].dt.to_period('Q').dt.to_timestamp('Q')
# Group by date and count renewals and cancellations
result = df.groupby(['end_date', 'subscription_type'])['renewed'].value_counts().unstack(fill_value=0).reset_index()
# Rename columns for clarity
result.columns = ['date', 'subscription_type','cancellations', 'renewed']
result['renewed_ratio'] = result['renewed'] / (result['cancellations'] + result['renewed'])
result['non_renewed_ratio'] = result['cancellations'] / (result['cancellations'] + result['renewed'])

result2 = result[result['subscription_type']=='Monthly']
result1 = result[result['subscription_type']=='Yearly']

custom_date1 = '2019-06-30'  # First custom date
custom_date2 = '2020-06-30'  # Second custom date
result1 = add_avg_indicator_value_given_dates(custom_date1,custom_date2,result1)
result2 = add_avg_indicator_value_given_dates(custom_date1,custom_date2,result2)
plot_2_time_series("First shock",result1, 'date','analisis','yearly_renewal_ratio_mean',
                    result2, 'date','analisis','monthly_renewal_ratio_mean')

```


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

custom_date1 = '2021-09-30'  # First custom date
custom_date2 = '2022-12-31'  # Second custom date
result1 = add_avg_indicator_value_given_dates(custom_date1,custom_date2,result1)
result2 = add_avg_indicator_value_given_dates(custom_date1,custom_date2,result2)
plot_2_time_series("Second shock",result1, 'date','analisis','yearly_renewal_ratio_mean',
                    result2, 'date','analisis','monthly_renewal_ratio_mean')

```

So, this are very interesting plots to make an argument. If we plot the mean renewal ratio prior, between and post the economics shocks, we can wee a similar impact on both yearly and monthly suscriptions renewal ratio of around .10 to .20 (10 to 20 points).

We can see that at the end of the shock, the trends seem to return to their normal levels. In the second graph, the last point of the monthly ratio is so high because it consists of only one observation, similar to the first point in the first graph. My personal prediction is that with time, the trend will return to their prior shock level, just like the yearly series did.

Wrapping up:

So far we had discovered some things that we can integrate into our model and findings:

+ There is an economic shock affecting the ratios of renewal of subscription
+ Inflation seems to affect yearly subscriptions

So now we can build a model considering the feature columns and a fixed effects for the economic shock effect.

## Model with economic shock

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

economic_indicators_to_work = economic_indicators.copy()
economic_indicators_to_work = economic_indicators_to_work[['end_date']]


# Define multiple date ranges
start_date_1 = '2021-09-30'
end_date_1 = '2022-12-31'

start_date_2 = '2019-06-30'
end_date_2 = '2020-06-30'

# Convert 'end_date' to datetime if not already
economic_indicators_to_work['end_date'] = pd.to_datetime(economic_indicators_to_work['end_date'])

# Assign 1 if 'end_date' falls within either of the two ranges
economic_indicators_to_work['economic_shock'] = (
    economic_indicators_to_work['end_date'].between(start_date_1, end_date_1) |
    economic_indicators_to_work['end_date'].between(start_date_2, end_date_2)
).astype(int)



df = client_details.merge(subscription_records)
df['end_date'] = df['end_date'].dt.to_period('Q').dt.to_timestamp('Q')
df = df.merge(economic_indicators_to_work)

df.pop('client_id')
df.pop('end_date')
df.pop('start_date')
best_model, grid_search = preprocess_and_logistic(df, 'renewed')

```

Now we have a quite strong model.
Key Metrics:

    Precision: Indicates the proportion of correctly predicted renewed subscriptions to the total.
        False: 80% precision means out of all non renewd subscriptions, 80% were correct.
        True: 64% precision means out of all renewed subcriptions, 64% were correct.

    Recall: Measures the proportion of correctly predicted positive observations to all observations in the actual class.
        False: 44% recall means only 44% of the actual non renewd subscriptions were correctly identified.
        True: 90% recall means 90% of the actual renewd subscriptions were correctly identified.

        

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

feature_importance = extract_feature_importance(best_model, df)
feature_importance[feature_importance['Coefficient']!=0]
```


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

feature_importance[feature_importance['Coefficient']<0]

```

This model has a really good performance because it only has depth 1 variables (and its only using 4 variables + the economic shock). This allows the model to generalize, otherwise we will be overfitting. However just for curiosity i will retrain the model with all combinations of variables including the economic shock one to see how are they interacting in predictions.


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"
economic_indicators_to_work = economic_indicators.copy()
economic_indicators_to_work = economic_indicators_to_work[['end_date']]
# Define date range
start_date = '2021-09-30'
end_date = '2022-12-31' 
# Add dummy variable column
economic_indicators_to_work['economic_shock'] = economic_indicators_to_work['end_date'].between(start_date, end_date).astype(int)




# Define multiple date ranges
start_date_1 = '2021-09-30'
end_date_1 = '2022-12-31'

start_date_2 = '2019-06-30'
end_date_2 = '2020-06-30'

# Convert 'end_date' to datetime if not already
economic_indicators_to_work['end_date'] = pd.to_datetime(economic_indicators_to_work['end_date'])

# Assign 1 if 'end_date' falls within either of the two ranges
economic_indicators_to_work['economic_shock'] = (
    economic_indicators_to_work['end_date'].between(start_date_1, end_date_1) |
    economic_indicators_to_work['end_date'].between(start_date_2, end_date_2)
).astype(int)


df = client_details.merge(subscription_records)
df['end_date'] = df['end_date'].dt.to_period('Q').dt.to_timestamp('Q')
df = df.merge(economic_indicators_to_work)

df['subscription_type_location'] = df['subscription_type'] + '_' + df['location']
df['subscription_type_company_size'] = df['subscription_type'] + '_' + df['company_size']
df['subscription_type_industry'] = df['subscription_type'] + '_' + df['industry']

df['industry_location'] = df['industry'] + '_' + df['location']
df['industry_company_size'] = df['industry'] + '_' + df['company_size']
df['location_company_size'] = df['location']  + '_' + df['company_size']

df['economic_shock_location'] = df['economic_shock'].astype(str) + '_' + df['location']
df['economic_shock_company_size'] = df['economic_shock'].astype(str)  + '_' + df['company_size']
df['economic_shock_industry'] = df['economic_shock'].astype(str)  + '_' + df['industry']

df.pop('client_id')
df.pop('end_date')
df.pop('start_date')
best_model, grid_search = preprocess_and_logistic(df, 'renewed')
feature_importance = extract_feature_importance(best_model, df)

feature_importance[feature_importance['Coefficient']>1]


```
In the last table we are filtering predictors with a positive impact on renewal. We can see here similar features like the one we did in the prior analysis but also new ones like that E commerce thrives in the absent of economic shocks.

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"
feature_importance[feature_importance['Coefficient']<0]

```
The last table shows features with negative impact on predicting a renewal. Again here we see mostly the same story than the analysis without the series, however in predicting non subscriptions, we can see the importance of the economic shock variable. Large, E commerce, Connecticut based and small companies were specially affected in the economic shock time.

## Dynamic table

Now to proove coherence with this automatic findings, we will return to our ratio analysis but now we are going to split the table of ratios by year and by economic shock:

### by Year

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

df = client_details.merge(subscription_records)
# level 2
df['subscription_type_location'] = df['subscription_type'] + '_' + df['location']
df['subscription_type_company_size'] = df['subscription_type'] + '_' + df['company_size']
df['subscription_type_industry'] = df['subscription_type'] + '_' + df['industry']
df['industry_location'] = df['industry'] + '_' + df['location']
df['industry_company_size'] = df['industry'] + '_' + df['company_size']
df['location_company_size'] = df['location']  + '_' + df['company_size']

# level 3
df['location_company_size_industry'] = df['location']  + '_' + df['company_size'] + '_' + df['industry']
df['location_company_size_industry_subscription_type'] = df['location']  + '_' + df['company_size'] + '_' + df['industry'] +df['subscription_type']
df['year'] = df['end_date'].dt.year

historic_df = []
for year in df.year.unique():
    whole_story = get_complete_story(df[df['year']==year])
    whole_story['year'] = year
    whole_story = whole_story[(whole_story['ratio_true'] > 0.6)& (whole_story['n'] > 5)]
    whole_story['ratio_true'] = whole_story["ratio_true"].astype(str).str[:5]

    historic_df = historic_df + [whole_story]
historic_df = pd.concat(historic_df)
create_dinamic_table(historic_df,"Top renewals profiles in")

```


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

historic_df = []
for year in df.year.unique():
    whole_story = get_complete_story(df[df['year']==year])
    whole_story['year'] = year
    whole_story = whole_story[(whole_story['ratio_true'] < 0.4)& (whole_story['n'] > 5)]
    whole_story['ratio_true'] = whole_story["ratio_true"].astype(str).str[:5]

    historic_df = historic_df + [whole_story]
historic_df = pd.concat(historic_df)
create_dinamic_table(historic_df ,"Top non renewals profiles in")

```

### by Economic Shock



```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

df = client_details.merge(subscription_records)
# level 2
df['subscription_type_location'] = df['subscription_type'] + '_' + df['location']
df['subscription_type_company_size'] = df['subscription_type'] + '_' + df['company_size']
df['subscription_type_industry'] = df['subscription_type'] + '_' + df['industry']
df['industry_location'] = df['industry'] + '_' + df['location']
df['industry_company_size'] = df['industry'] + '_' + df['company_size']
df['location_company_size'] = df['location']  + '_' + df['company_size']

# level 3
df['location_company_size_industry'] = df['location']  + '_' + df['company_size'] + '_' + df['industry']
df['location_company_size_industry_subscription_type'] = df['location']  + '_' + df['company_size'] + '_' + df['industry'] +df['subscription_type']
df['year'] = df['end_date'].dt.year

# Define multiple date ranges
start_date_1 = '2021-09-30'
end_date_1 = '2022-12-31'

start_date_2 = '2019-06-30'
end_date_2 = '2020-06-30'

# Convert 'end_date' to datetime if not already

# Assign 1 if 'end_date' falls within either of the two ranges
df['economic_shock'] = (
    df['end_date'].between(start_date_1, end_date_1) |
    df['end_date'].between(start_date_2, end_date_2)
).astype(int)

historic_df = []
for year in df.economic_shock.unique():
    whole_story = get_complete_story(df[df['economic_shock']==year])
    whole_story['year'] = year
    whole_story = whole_story[(whole_story['ratio_true'] > 0.65)& (whole_story['n'] > 5)]
    whole_story['ratio_true'] = whole_story["ratio_true"].astype(str).str[:5]

    historic_df = historic_df + [whole_story]
historic_df = pd.concat(historic_df)
create_dinamic_table(historic_df,"Top renewals profiles in economic shock = ")


```



```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

historic_df = []
for year in df.economic_shock.unique():
    whole_story = get_complete_story(df[df['economic_shock']==year])
    whole_story['year'] = year
    whole_story = whole_story[(whole_story['ratio_true'] < 0.45)& (whole_story['n'] > 5)]
    whole_story['ratio_true'] = whole_story["ratio_true"].astype(str).str[:5]
    historic_df = historic_df + [whole_story]
historic_df = pd.concat(historic_df)
create_dinamic_table(historic_df,"Top non renewals profiles in economic shock = ")
```

This last exercie confirms what the automatic features shown: in predicting non subscriptions, we can see the importance of the economic shock variable. Companies with yearly subscriptions, Large, E commerce reltated, Connecticut based were specially affected in the economic shock time. 


+ Cryto and large companies are not renewing in general in shock and non shock times. 
+ Despite already having low ratios, yearly subscriptions were specially affected during shocks.
+ Mothly Crypto

+ Gaming renewals were high in non shock times
+ Medium and AI companies were reselient and had high ratios of renewals even during economic shocks.
