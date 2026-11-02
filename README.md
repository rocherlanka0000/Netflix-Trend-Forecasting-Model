# Task 5 – Netflix Trend Forecasting Model

Time series forecasting: predict how many titles Netflix adds to its catalogue each month, from historical data.

## Step 1: Prepare time-based features
- Target: **number of titles added per month** (`date_added`), the only column with monthly resolution. (`release_year` goes back to 1925 and describes production, not Netflix trends.)
- Series used: **January 2016 to August 2021** (68 months). Before 2016 there are only a few titles per year, and September 2021 is incomplete (data stops on 25 September).
- Features for the machine learning models: trend index, month (sine / cosine), lags (1, 2, 3 and 12 months), rolling means (3 and 6 months). Only past values are used, so there is no leakage from the future.

## Step 2: Analyze historical release patterns
- Strong growth from 2016 to 2019 (426, 1,185, 1,647, 2,014 titles), then a plateau: about 168 titles per month in 2019, 157 in 2020 and 164 in January to August 2021.
- Moderate yearly pattern (strength 0.35): busy in December, October and March; quiet in February, May and June.
- Analysis tools: year x month heatmap, monthly boxplot, seasonal decomposition, ADF stationarity test, autocorrelation plot, Movies vs TV Shows trend.

## Step 3: Build forecasting models
Eight models, so that complex ones must beat simple baselines:
- **Baselines**: Naive, Seasonal naive, Mean of last 6 months
- **Statistical**: Holt (damped trend), Holt-Winters, SARIMA(1,1,1)(0,1,1,12)
- **Machine learning**: Random Forest and Gradient Boosting on lag features (recursive multi-step forecasting)

## Step 4: Generate future predictions
The best model of the rolling backtest (Random Forest) is refit on all data and forecasts the next 12 months (September 2021 to August 2022), with an 80% prediction interval built from its backtest errors.

| | Titles |
|---|---|
| Last 12 months (actual) | 1,973 (164 per month) |
| Next 12 months (forecast) | 1,844 (154 per month), **-6.5%** |
| 80% interval, one month | about 40 to 50 titles on each side |

Differences between forecast months (141 to 167) are much smaller than the uncertainty, so single "peak" months should not be over-interpreted.

## Step 5: Evaluate forecasting accuracy
Always training on the past and testing on the future, in two ways: a 12-month holdout, and a rolling-origin backtest (15 starting points, 3-month horizon). Metrics: MAE, RMSE, MAPE, MASE.

| Model | MAE, 12-month holdout | MAE, rolling backtest |
|---|---|---|
| Random Forest | 34.4 | **29.9** |
| Mean of last 6 months | 33.5 | 30.6 |
| Naive | 41.6 | 34.2 |
| Gradient Boosting | 42.9 | 34.5 |
| Holt (damped trend) | **31.8** | 34.7 |
| Holt-Winters | 40.1 | 35.0 |
| Seasonal naive | 47.3 | 37.2 |
| SARIMA | 46.7 | 37.9 |

MAE is the average error in titles per month (lower is better). The series is short and noisy, so **simple models are hard to beat**, and the winner changes between the two evaluations. That is why both are reported.

## Limitations
- `date_added` only includes titles still on Netflix when the dataset was made, so older months are under-counted.
- 2020 and 2021 were affected by the pandemic, which the models cannot anticipate.
- The data stops in September 2021, so "future" means the 12 months after the end of the dataset.

## Run the project
```bash
pip install -r requirements.txt
jupyter notebook task5_trend_forecasting.ipynb
```
