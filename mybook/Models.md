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

# Automatic anaylisis - Models

This was a digestible analysis because the dataset is of size 100. However in real life we might face bigger datasets and this kind of analysis will be harder to make. In this part im going to replicate the analysis but with a big data - ML model approach. With a model for analysis, we can detect hidden trents in an automatic way, and we can trust the results because the way the model is build is actually very similar to what we did manually before.

We are constrained with the size of the dataset, so the model metrics are not going to be the best (because with n= 100 and level 2 variables we are loosing degrees of freedom) but the results on the feautures (combinations) to select should be similar.

We can proposed any model we want (a random forest, a lasso or ridge regression) but i consider that a logistic regression could be the best model considering that for now we only have categorical data.


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


#Automatic analyss with depth level 2 variables
df = client_details.merge(subscription_records)
df['subscription_type_location'] = df['subscription_type'] + '_' + df['location']
df['subscription_type_company_size'] = df['subscription_type'] + '_' + df['company_size']
df['subscription_type_industry'] = df['subscription_type'] + '_' + df['industry']

df['industry_location'] = df['industry'] + '_' + df['location']
df['industry_company_size'] = df['industry'] + '_' + df['company_size']
df['location_company_size'] = df['location']  + '_' + df['company_size']
df.pop('client_id')
df.pop('end_date')
df.pop('start_date')
#df['renewed'] = ~df['renewed']


best_model, grid_search = preprocess_and_logistic(df, 'renewed')

```


So our model was built to predict who actually is renewing however it has a mediocre predictive capacity bus mostly due to the dataset size, remember that this tools are designed to thrive with big data, however they still are usefull to see the importance of the features. Lets take a look on which features and how much are they contributing for predictions.

So in the next table we are filtering for the 15 best features that contribute in a negative way to predict if the company is going to renew. This means that this features are profiling companies that are not going to renew. 

We can see similar results:
+ The best predictor for a non renewing company is if it is a large company with a yearly suscription.
+ Also companies with labels like large and crypto are not renewing.
+ We see ome familiar profiles like small fintech and yearly Massachusetts




```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"
feature_importance = extract_feature_importance(best_model, df)
feature_importance[feature_importance['Coefficient']<0][0:15]
```

On the other side of the coin we see very similar predictors for who is renewing:
+ Monthly suscriptors, Gaming companies, Medium size companies, Ai comanies etc

Note that the predictors have a bigger absolute coefficient, this means that they contribute more to the predictions that the predictors from the last chart. Its easier to identify who is renewing.

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"
feature_importance[feature_importance['Coefficient']>0][0:15]

```

This models are usefull for the moment just to see the importance of features but we cant give them a strict interpretaion because the significance is minimal due to the size of the dataset and the amount of features.

```{seealso}
# 📌 Interpretation of Parameters in a Logistic Regression

Logistic regression models the **probability of an event occurring** (e.g., `renewal = 1`) as:

$$\log\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n$$

Where:
- $( p )$ = Probability of the event occurring.
- $( X_1, X_2, ..., X_n )$ = Independent variables (predictors).
- $( \beta_0 )$ = Intercept.
- $( \beta_1, \beta_2, ... )$ = Coefficients.

## 🔹 Interpretation of Coefficients

- **Positive coefficient $(\beta > 0)$** → An **increase in the variable** increases the probability of the event occurring.
- **Negative coefficient $(\beta < 0)$** → An **increase in the variable** decreases the probability of the event occurring.
- **$(\beta \approx 0)$** → The variable has **little or no effect** on the outcome.

### 🔹 Odds Ratio
The odds ratio $(\text{OR}\)$ is given by:

$$\text{OR} = e^{\beta}$$

- **If OR > 1** → The variable **increases** the odds of the event.
- **If OR < 1** → The variable **decreases** the odds of the event.
- **If OR = 1** → The variable has **no effect**.

✅ **Example Interpretation**  
- If **$(\beta = 0.000004)$** → $( e^{0.000004}$ approx 2.01 \)  
  → **A 1-unit increase in the variable doubles the odds** of the event occurring.
- If **$(\beta = -0.5)$** → $( e^{-0.5}$ approx 0.61 \)  
  → **A 1-unit increase in the variable reduces the odds by ~39%**.

---

## 🔹 Summary Table

| Coefficient (\(\beta\))  | Interpretation |
|----------------|------------------------------------------------|
| **$(\beta > 0)$** | Higher values **increase** probability of the event. |
| **$(\beta < 0)$** | Higher values **decrease** probability of the event. |
| **$(\beta \approx 0)$** | The variable has **little or no effect**. |
| **$( e^\beta )$ (Odds Ratio)** | If **>1**, increases event probability; if **<1**, decreases it. |

---

```