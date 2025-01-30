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
# Graphical analysis

## Renewals trends

We now have a clearer picture of who is renewing and who is not renewing. However lets check how are renewals behaving across time.




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

plot_2_time_series("Comparison between renewals ratios by quarter",result1, 'date','renewed_ratio','renewed_ratio_monthly',
                    result2, 'date','renewed_ratio','renewed_ratio_yrl')

```
What we can observe from this time series is that yearly and monthly renwals actually behave differently. Although both have a cyclical pattern, it seems that the monthly renewals ratio has a wider range, this means that its more volatile. On the contrast, the yearly renewals ratios seems to be more stable. Just look at the trend from 2020 to 2023 where the ratio is constrained between the 0 and .7values. In general, the average yearly suscriptions renew ratio seems to be smaller than the monthly but the monthly had a wider range.

Also we can see a stable pattern before the half of 2021 and then the ratio went down consistently, possible due to a negative correlated scenario between gdp growth and inflation.


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

result2['year'] = result2['date'].dt.year
result1['year'] = result1['date'].dt.year

# Group by year and compute the mean for each economic indicator
result1_summary = result1.groupby('year').agg({
    'renewed_ratio': 'mean',
}).reset_index()

# Group by year and compute the mean for each economic indicator
result2_summary = result2.groupby('year').agg({
    'renewed_ratio': 'mean',
}).reset_index()


plot_2_time_series("Comparison between renewals ratios by year",result1_summary, 'year','renewed_ratio','yearly_renewal_ratio_mean',
                    result2_summary, 'year','renewed_ratio','monthly_renewal_ratio_mean')

```

This second graph is quite usefull because it is showing that yes, there is like a different intercept (starting point or fixed effect) for each ratio but the slopes are quite similar for each year, or at least they go in the same direction. 

We can conclude from this section that 

+ yearly and monthly renewals behave similarly if we grouped them by year: they seem to have a different fixed effect (yearly subscriptions have a lower rate of renewal) however they changed in the same direction during the scope of the data.

+ on average, both trends grew during 2018, 2020 and probably 2023 (we dont have full data); and both trends shrinked during 2019 and 2021.

## Economic indicators

We now have a clearer picture of who is renewing and who is not renewing and how renewals are behaving across time. However lets check how are the external economical trends impacting. Now we are going to explore the economic indicators.

The first step is just to see the trends, propose some hypothesis backed by macroeconomical theory and test them. So the first thing is to see the economic series.


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"


plot_2_time_series("Comparison beetween econonic series",economic_indicators, 'end_date','inflation_rate','inflation_rate',
                    economic_indicators, 'end_date','gdp_growth_rate','gdp_growth_rate')


```

What we can see here is that the gdp growth series is more stable than the inflation rate. Also, from a first glance, we can see 2 different periods: the first one before half of 2021 where we can see both indicators very correlated and a second period where both indicators almost went in the opposite directions (negative correlation).

Now lets take a deeper look: we want to see the times where both trends have a possitive or negative correlation. We can achieve this by symply calculating the difference for each indicator between each quarter. If we multiply this differential, we would get a variable that is possitive when both differentials have the same sign, and negative when both are different. This means that our indicator is a proxy of correlation.

```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"

economic_indicators["inflation_change"] = economic_indicators["inflation_rate"].diff()
economic_indicators["gdp_growth_change"] = economic_indicators["gdp_growth_rate"].diff()
economic_indicators["cor"] = economic_indicators["inflation_change"]  * economic_indicators["gdp_growth_change"] 

plot_1_time_series_with_line(economic_indicators, 'end_date','cor','differentials product', -5)


```

```{seealso}


Just a macroeconomics reminder:

When GDP growth and inflation are correlated, it means there is a statistical relationship between the two variables—changes in one tend to be associated with changes in the other. The degree and direction of this relationship are indicated by the correlation coefficient:

    Positive correlation: When GDP growth increases, inflation also tends to increase, and when GDP growth decreases, inflation tends to decrease.
        Example: In periods of strong economic growth, demand for goods and services often rises, which can push prices up, leading to higher inflation.

    Negative correlation: When GDP growth increases, inflation tends to decrease, and vice versa.
        Example: In some cases, rapid GDP growth might coincide with efficiency gains or deflationary pressures (like technological advancements), reducing inflation. Conversely, during a slowdown or recession, inflation could rise due to supply-side shocks (e.g., higher input costs like energy).

A negative correlation between GDP growth and inflation means that when GDP growth improves, inflation generally declines, and when GDP growth slows, inflation increases.

This could indicate scenarios such as:

    Supply-side constraints: If the economy slows because of higher production costs (e.g., oil price spikes), inflation might rise due to cost-push inflation.
    Monetary policy effects: Central banks might aggressively combat inflation (e.g., raising interest rates), leading to slower economic growth.
    Deflationary growth: Rare situations where economic growth comes with falling prices, often driven by technological progress or globalization.
```

So by looking at the trends, we can differentiate the periods where the differentials products got very low values (below an arbitrary benchmark). We are going to identify this 2 moments as an external economic shock.

If we plot this differential correlation with the yearly renewal ratio we can see a high correlation in this 2 periods of 2019-2020 and 2021-2022, where the renewals yearly ratio almost went to zero. 


```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"
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

experiment = scale_columns(economic_indicators,['cor'])

plot_2_time_series("Macroeconomi indicator vs yearly renewed ratio",experiment, 'end_date','cor','differentials product',
                    result1, 'date','renewed_ratio','renewed_ratio_yrl')

```
This finding is coherent with our conclusions of the last chapter: **"...on average, both trends grew during 2018, 2020 and probably 2023 (we dont have full data); and both trends shrinked during 2019 and 2021."**

## Raw correlations

Finally, lets plot the economic series along with our ratios but with some transformations.

+ First lets put them in a similar vector space: the range 0-1, to make it easier to the eye (normalization).
+ Then lets create the non renewal ratio: this because by intuition we suppose that higher inflation might affect the renewals negatively.



```{code-cell}
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show code"
:  code_prompt_hide: "Hide code"
experiment = scale_columns(economic_indicators,['inflation_rate','gdp_growth_rate'])
plot_2_time_series("Inflation vs non renewals",experiment, 'end_date','inflation_rate','inflation_rate',
                    result1, 'date','non_renewed_ratio','non_renewed_ratio_yrl')
```
We can see here a positive correlation between inflation rate and yearly non renewals. Which makes economic sense: companies are facing inflationary pressures and some of them decided to non renew.

## Summary

+ yearly and monthly renewals behave similarly if we grouped them by year: they seem to have a different fixed effect (yearly subscriptions have a lower rate of renewal) however they changed in the same direction during the scope of the data.

+ on average, both trends grew during 2018, 2020 and probably 2023 (we dont have full data); and both trends shrinked **during 2019 and 2021**.

+ we detected an economic shock **during 2019 and 2021**.

+ inflations is correlated with yearly non renewals.