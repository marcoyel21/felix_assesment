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
cell_metadata_filter: -all
---
# Insights

## Summary

With our frequency analysis, time series and machine learning models we were able to create a picture of which companies are renewing which are not and how are external economic factors impact this decision.

In general:

+ Companies with yearly subscription and companies in crypto industry are not renweing as much as the rest of the companies.
+ Companies with monthly subscription, medium size and in gaming and AI industries tend to renew more than the rest.
+ Inflation is highly correlated with the non renewing ratio of companies with yearly subcriptions.
+ We see 2 economic moments with 2 different trends: during economic contraction and during economical normality.
+ During the economic contraction, inflation and economic growth had a negative correlation indicating possible slowdown or recession. 
+ This affected the general trends of renewals in a negative form.
+ The effect had a special impact on certain companies: Large, E commerce, Connecticut based and small companies.

In specific:

+ Large companies with yearly suscriptions from all the country and all industries didnt renewed 80% of the time. This was the case for 12 of the 15 cases. **This is very important because this explains more than half of the yearly suscriptions that got canceled (12/23)**
+ Crypto companies of size large from any location didnt renewed 70% of the time. This was the case for 7 of the 10 cases.
+ Companies from Massachusets for any industry with a yearly suscription didnt renewed 71% of the time. This was the case for 5 of the 7 cases.
+ Small fintechs from all locations didnt renewed 71% of the time. This was the case for 5 of the 7 cases.
+ **Small gaming companies have renewd 100% of the time.**
+ Pennsylvania medium companies renew 87% of the time.
+ AI companies with monthly subscriptions renew 83% of the time.
+ Gaming companies with monthly subscriptions renew 83% of the time.
+ Fintech medium size companies renew 75% of the time.. **This is interesting because small fintechs are not renewing.**
+ Massachusetts small companies renew 75% of the time.
+ Gaming New Jersey companies renew	75%.
+ **Gaming companies renew 72% of the time. This account for almost a third of all renewals (16/55)**
+ The economic shock affected the general trends of renewals in a negative form: the ratios of both monthly and yearly subcriptions declined during this shock in around 10 to 20 points.
+ Large, E commerce, Connecticut based and small companies were specially affected during the economic shock.

## Recomendations

A quadrant analysis is going to help us visualize the data and create intutive general strategies. In the next graphis you can see for each profile of companies their ratio of renewals during economic contraction (X) and during good times (Y). Each profile is represented by a dot with a size proporcional to the number of companies within.

+ Always high ratio: here are the companies that are renewing every year and that are resilient to economic contractions. **This companies do not need urgent attention and we can rely on their renewals.**

+ Always low ratio: here are the companies that are not renewing even in good times. **This companies need urgent and specific attention.**

+ Impacted deeply by shock: here are the companies that were renewing during good times but were affected by the contraction. **This companies need special attention only during economic shocks so we can build a follow up strategy.**

+ Helped by shock: are outliers companies whoose renewals increased during the contraction. They are only a few profiles and mostly are small companies so it could be that this phenomenon was just statistical noise, but **we dont have enough data to proove that.**

In the next graph you can see each profile with the cursor.

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
df = client_details.merge(subscription_records)# level 2
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

ws_1 = get_complete_story(df[df['economic_shock']==1])
ws_1['n_1'] = ws_1['n']
ws_1['ratio_true_shock'] = ws_1['ratio_true']
ws_1 = ws_1[['indicator','n_1','ratio_true_shock']]
ws_0 = get_complete_story(df[df['economic_shock']==0])
ws_0['n_0'] = ws_0['n']
ws_0['ratio_true_no_shock'] = ws_0['ratio_true']
ws_0 = ws_0[['indicator','n_0','ratio_true_no_shock']]

d = ws_1.merge(ws_0)

d['n']=d['n_0'] +d['n_1']



import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame (Replace with actual data)
df = d
# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 7))

# Scatter plot with size proportional to 'n'
scatter = ax.scatter(df["ratio_true_no_shock"], df["ratio_true_shock"], 
                     s=df["n"], color="blue", alpha=0.6, edgecolors="black", label="Data Points")

# Center at (0.5,0.5) by adding dashed lines
ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1)  # Horizontal center line
ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1)  # Vertical center line

# Set axis labels
ax.set_xlabel("Ratio During No Shock", fontsize=12)
ax.set_ylabel("Ratio During Shock", fontsize=12)

# Set grid and limits
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, linestyle=":", alpha=0.5)

# Add quadrant labels
ax.text(0.25, 0.75, "Helped by shock", fontsize=12, ha="center", color="darkred")
ax.text(0.75, 0.75, "Always high ratio", fontsize=12, ha="center", color="darkgreen")
ax.text(0.25, 0.25, "Always low ratio", fontsize=12, ha="center", color="darkblue")
ax.text(0.75, 0.25, "Impacted deeply by shock", fontsize=12, ha="center", color="purple")

# Add title
plt.title("4-Quadrant Analysis of Renewal Ratios", fontsize=14)

# Show plot
plt.show()


```

F

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

import pandas as pd
import plotly.express as px

# Sample DataFrame (Replace with actual data)
df = d

# Create the scatter plot using Plotly
fig = px.scatter(
    df, 
    x="ratio_true_no_shock", 
    y="ratio_true_shock", 
    size="n",  # Bubble size
    hover_name="indicator",  # Show indicator name on hover
   # text="indicator",  # Optional: Show text labels directly
    labels={"ratio_true_no_shock": "Ratio During No Shock", "ratio_true_shock": "Ratio During Shock"},
    title="4-Quadrant Analysis of Renewal Ratios"
)

# Add quadrant lines at (0.5,0.5)
fig.add_hline(y=0.5, line_dash="dash", line_color="red")
fig.add_vline(x=0.5, line_dash="dash", line_color="red")

# Show the interactive plot
fig.show()
```


## Further discussion

The anaylisis is limited by the dataset size and personal time but could be improved with:
+ more data
+ hypothesis testing for the difference in means for the shock vs no shock mean analysis.
+ a better model building with significance tests on parameters
k.
+ Proper lenguage depending on the audience (in general better enlight, more or less technicalities)
