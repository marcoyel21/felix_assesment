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
cell_metadata_filter: -all

---

# Exploratory Data Analysis

## Ratio analysis

For this we are going to simply count the ratio of non renewals for each category. But, we are going to make custom categories based on combinations. The first level of combinations are our columns; the second level are the combination between each 2 of them; the third level is the combination of each 3 of them.

What we want to see is for each combination possible, the ratio of $ renewd/(renewed + non renewed)$, then sort them based on the higher and lower ratios and the higher number of occurrences of each combination.


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

import pandas as pd
from src.utils import *

client_details = pd.read_csv('data/client_details.csv')
subscription_records = pd.read_csv('data/subscription_records.csv', parse_dates = ['start_date','end_date'])
economic_indicators = pd.read_csv('data/economic_indicators.csv', parse_dates = ['start_date','end_date'])

df = client_details.merge(subscription_records)
whole_story = create_whole_story_df(df)
whole_story[(whole_story['ratio_true'] > 0.6)& (whole_story['n'] > 5)]

```

We build a table with all the combinations possible, their ratio, the level (depth) of combinations and the occurrences (n). We then filter only for those combinations where the ratio of renewal is above .6 (60%) and combinations that happened more than 5 times (which is a fair number considering the dataset is of size 100).

The first thing we note is that we have more combinations with at least 5 occurcences and that we have even level 1 profiles (just one column). This means at a glance that its actually easy to identify who is reneweng.

In general we can see that profiles with monthly, medium, gaming and AI labels tend to renew.

Also we have identified some specific profiles:
+ **Small gaming companies have renewd 100% of the time.**
+ Pennsylvania medium companies 87% of the time.
+ Monthly AI 83%.
+ Monthly gaming 80%.
+ Fintech medium 75%. This is interesting because small fintechs are not renewing as you will see in the next table.
+ Massachusetts small 75%
+ Gaming New Jersey	75%
+ **All gaming 72%. This account for almost a third of all renewals (16/55)**
+ among others

Now we are going to see the other side of the coin, who is renewing? So we take the same ratios table and filter for combinations with a lower ratio of .4. This are profiles that are renewend more than 60% of the time.

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

whole_story[(whole_story['ratio_true'] < 0.4) & (whole_story['n'] > 5)].sort_values(by='ratio_true', ascending=True)
```

We can see in the table that:
+ It seems that combinations around yearly and crypto are not renweing.
+ We only need combinations of level 2 to have a good hypothesis of whats happening.

Also we have identified specific profiles:
+ Large companies with yearly suscriptions from all the country and all industries didnt renewed 80% of the time. This was the case for 12 of the 15 cases. **This is very important because this explains more than half of the yearly suscriptions that got canceled (12/23)**
+ Crypto companies of size large from any location didnt renewed 70% of the time. This was the case for 7 of the 10 cases.
+ Companies from Massachusets for any industry with a yearly suscription didnt renewed 71% of the time. This was the case for 5 of the 7 cases.
+ Small fintechs from all locations didnt renewed 71% of the time. This was the case for 5 of the 7 cases.

## Summary

So in summary we can see a more detailed story:

+ It seems that combinations around yearly and crypto are not renweing.
+ In general we can see that profiles with monthly, medium, gaming and AI labels tend to renew.
+ We only need combinations of level 2 to have a good hypothesis of whats happening.
+ Large companies with yearly suscriptions from all the country and all industries didnt renewed 80% of the time. This was the case for 12 of the 15 cases. **This is very important because this explains more than half of the yearly suscriptions that got canceled (12/23)**
+ **All gaming has a renewal rate of72%. This account for almost a third of all renewals (16/55)**
+ **Small gaming companies have renewd 100% of the time.**