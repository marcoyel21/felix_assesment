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

# The Data

The company has provided us with three datasets for our analysis. A summary of each data is provided below.

## `client_details.csv`

| Column         | Description|
|----------------|---------------------------------------------------------------|
| `client_id`    | Unique identifier for each client. |
| `company_size` | Size of the company (Small, Medium, Large).|
| `industry`     | Industry to which the client belongs (Fintech, Gaming, Crypto, AI, E-commerce).|
| `location`     | Location of the client (New York, New Jersey, Pennsylvania, Massachusetts, Connecticut).|

## `subscription_records.csv`

| Column             | Description   |
|--------------------|---------------|
| `client_id`        | Unique identifier for each client.|
| `subscription_type`| Type of subscription (Yearly, Monthly).|
| `start_date`       | Start date of the subscription - YYYY-MM-DD.|
| `end_date`         | End date of the subscription - YYYY-MM-DD.|
| `renewed`          | Indicates whether the subscription was renewed (True, False).|

## `economic_indicators.csv`

| Column           | Description                                       |
|------------------|---------------------------------------------------|
| `start_date`     | Start date of the economic indicator (Quarterly) - YYYY-MM-DD.|
| `end_date`       | End date of the economic indicator (Quarterly) - YYYY-MM-DD.|
| `inflation_rate` | Inflation rate in the period.|
| `gdp_growth_rate`| Gross Domestic Product (GDP) growth rate in the period.|


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

import pandas as pd
#from src.utils import *

client_details = pd.read_csv('data/client_details.csv')
subscription_records = pd.read_csv('data/subscription_records.csv', parse_dates = ['start_date','end_date'])
economic_indicators = pd.read_csv('data/economic_indicators.csv', parse_dates = ['start_date','end_date'])
df = client_details.merge(subscription_records)
```

The first thing we are going to do is just explore the dataset by each factor column. For company size, we see a balanced renewed count for the categories except for medium size where renewals dominate.

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

df.groupby(['company_size', 'renewed']).size().reset_index(name='count')
```


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

df.groupby(['subscription_type', 'renewed']).size().reset_index(name='count')
```

For suscription type we see that yearly suscripcion almost have 50% probability of being renewd whereas monthly have a slightly bigger one (35/57 = 61%).

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

df.groupby(['industry', 'renewed']).size().reset_index(name='count')
```

Regarding industry we can see that AI and Gaming have higher renewed ratios (7/11 and 16/22). Whereas the other 3 industries have a more balanced ratio with crypto industry renewed ratio being the lowest (11/25)

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

df.groupby(['location', 'renewed']).size().reset_index(name='count')
```

In location we can see that all locations seem kind of balanced around a .5 ratio or 50% probability, except for New York (58%) and Pennsylvania (60%) that have a slightly bigger tendency to renew.

So far we can see some big picture trends: crypto industries are not being renewed that often; monthly suscriptions, ai and gaming industries and New York and Pennsylvania companies are renewing quite often. The rest of the factors seem quite balanced.