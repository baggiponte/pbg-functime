import marimo

__generated_with = "0.1.71"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Introduction to time series forecasting with functime

        ![image.png](https://github.com/descendant-ai/functime/raw/main/docs/img/banner_dark_bg.png)

        [functime](functime.ai) is a lightweight time series library that enables scaling forecasts of large panel datasets effortless, thanks to Polars and global forecasting.

        In other words: with functime, you can **train forecasting models on thousands of time series on your laptop** (no PySpark required!), and feature engineering is blazingly fast thanks to Polars' parallelisation capabilities.

        ## Data loading

        The dataset contains monthly observations of some 70 commodities, between 1960 and 2023. The data has a monthly frequency. Our goal is to forecast the price of the next quarter.
        """
    )
    return


@app.cell
def __():
    import polars as pl

    url = "https://github.com/functime-org/functime/raw/main/data/commodities.parquet"

    raw = pl.scan_parquet(url)
    return pl, raw, url


@app.cell
def __(pl, raw):
    freq = "1mo"
    forecasting_horizon = 4  # a quarter

    entity, time, target = tuple(pl.col(col) for col in raw.columns)
    return entity, forecasting_horizon, freq, target, time


@app.cell
def __(pl, raw, target, time):
    y = raw.with_columns(
        time.cast(pl.Date),
        target.shrink_dtype(),
    )

    y.fetch(5)
    return y,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        functime only makes one assumption about the data:

        1. The first column identifies the time series (this is usually referred to as *entity*).
        2. The second column is the datetime index.
        3. The third one is the target.

        There can also be additional covariates, such as holiday effects or macroeconomic variables.

        > RFC: Do you like these defaults? What do you think of a `PanelDataFrame` class that inherits from a `polars.DataFrame`, whose constructor allows you to specify the names of the entity, time and target cols?

        # Exploratory Data Analysis

        It would be impossible to make sense of 70 time series, one at a time. For this reason, plotting utilities play a central role in functime. For example, we can use the `plotting.plot_entities` function to display the number of observations in each series.

        The number of observations per entity is not constant: as we can see from the plot below, the number ranges from approximately 250 to 750.
        """
    )
    return


@app.cell
def __(y):
    from functime import plotting

    plotting.plot_entities(y)
    return plotting,


@app.cell
def __(entity, plotting, y):
    # this will become a feature of `plotting.plot_{panel,forecasts,backtests}` from the next release
    # PR is just waiting to be merged 🔥
    k = 6

    commodities = y.select(entity.unique(maintain_order=True))

    sample = commodities.select(entity.sample(k, seed=42)).collect().to_series()

    (
        y.filter(entity.is_in(sample)).pipe(
            plotting.plot_panel,
            title=f"Sample of {k} time series",
        )
    )
    return commodities, k, sample


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Feature Extraction

        To make sense of many time series, we cannot simply plot a subsample. For this reason, functime offers two incredible sets of features:

        1. **Feature extractors**. functime registers a Polars namespace called `.ts`. In other words, as soon as you `import functime`, you can call `pl.col(...).ts.` and access blazingly fast feature extractors.
        2. **Ranking functions** to sort the data by a metric of choice.
        """
    )
    return


@app.cell
def __(entity, k, plotting, target, y):
    bottom_k = (
        y.group_by(entity)
        .agg(target.ts.variation_coefficient().alias("cv"))
        .bottom_k(k=k, by="cv")
        .select(entity)
        .collect()
        .to_series()
    )

    (
        y.filter(entity.is_in(bottom_k)).pipe(
            plotting.plot_panel,
            title=f"{k} series with the lowest coefficient of variation",
        )
    )
    return bottom_k,


@app.cell
def __(entity, k, plotting, target, y):
    # just opened an issue about this: https://github.com/functime-org/functime/issues/165

    top_k = (
        y.group_by(entity)
        .agg(target.ts.variation_coefficient().alias("cv"))
        .top_k(k=k, by="cv")
        .select(entity)
        .collect()
        .to_series()
    )

    (
        y.filter(entity.is_in(top_k)).pipe(
            plotting.plot_panel,
            title=f"{k} series with the highest coefficient of variation",
        )
    )
    return top_k,


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Train-test split

        functime features the familiar scikit-learn API:
        """
    )
    return


@app.cell
def _(forecasting_horizon, y):
    from functime.cross_validation import train_test_split

    splitter = train_test_split(forecasting_horizon)
    y_train, y_test = splitter(y)
    return splitter, train_test_split, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"functime functions, however, are designed to work with panel data:")
    return


@app.cell
def _(y_test):
    y_test.head(12).collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Benchmark forecast

        Some forecasting methods are extremely simple and surprisingly effective. Naive and seasonal naive forecasters are surprisingly hard to beat! You should always consider using the naive and seasonal naive forecasts as benchmarks. The naive forecaster repeats the last value and, for a multi-step forecast, returns a flat line. The seasonal naive returns the same pattern of length `sp` (seasonal periodicity). [reference](https://otexts.com/fpp3/simple-methods.html#simple-methods)
        """
    )
    return


@app.cell
def _(freq):
    from functime.forecasting import snaive

    forecaster_naive = snaive(
        freq=freq,
        sp=12,
    )
    return forecaster_naive, snaive


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"Now we train the seasonal naive model:")
    return


@app.cell
def _(forecaster_naive, forecasting_horizon, y_train):
    _ = forecaster_naive.fit(y=y_train)
    y_bench_pred = forecaster_naive.predict(fh=forecasting_horizon)

    y_bench_pred.head(12)
    return y_bench_pred,


@app.cell
def _(entity, k, plotting, sample, y_bench_pred, y_test):
    plotting.plot_forecasts(
        y_test.filter(entity.is_in(sample)).collect(),
        y_bench_pred.filter(entity.is_in(sample)),
        title=f"Sample of {k} naive forecasts",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"Once again, how can we evaluate the goodness of fit across dozens of series? functime offers the `rank_*` functions:"
    )
    return


@app.cell
def _(y_bench_pred, y_test):
    from functime.evaluation import rank_point_forecasts

    rank_bench_point_forecast = rank_point_forecasts(
        y_true=y_test,
        y_pred=y_bench_pred,
        sort_by="mape",
        descending=True,
    )
    return rank_bench_point_forecast, rank_point_forecasts


@app.cell
def _(
    entity,
    k,
    plotting,
    rank_bench_point_forecast,
    y_bench_pred,
    y_test,
):
    worst_bench_point_forecast = (
        rank_bench_point_forecast.select(entity).to_series().head(k)
    )

    plotting.plot_forecasts(
        y_test.filter(entity.is_in(worst_bench_point_forecast)),
        y_bench_pred.filter(entity.is_in(worst_bench_point_forecast)),
        title=f"Worst {k} naive forecasts by mean absolute percentage error (MAPE)",
    )
    return worst_bench_point_forecast,


@app.cell
def _(
    entity,
    k,
    plotting,
    rank_bench_point_forecast,
    y_bench_pred,
    y_test,
):
    best_bench_point_forecast = (
        rank_bench_point_forecast.select(entity).to_series().tail(k)
    )

    plotting.plot_forecasts(
        y_test.filter(entity.is_in(best_bench_point_forecast)),
        y_bench_pred.filter(entity.is_in(best_bench_point_forecast)),
        title=f"Best {k} naive forecasts by mean absolute percentage error (MAPE)",
    )
    return best_bench_point_forecast,


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are, however, better approaches to glance the goodness of fit - which we shall see after we fit another model.

        # Global models

        What makes functime so fast is also the fact that only one model is fit on the whole dataset - hence why the name _global forecasting_. This strategy first proved successful with the [M5 competition](https://www.sciencedirect.com/science/article/pii/S0169207021001874?via%3Dihub), while there is a growing amount of [literature](https://www.sciencedirect.com/science/article/abs/pii/S0169207021000558?via%3Dihub) (mostly from Hyndman and his coauthors). This enables a ["new paradigm of forecasting"](https://blogs.sas.com/content/forecasting/2016/10/25/changing-the-paradigm-for-business-forecasting-part-10/):

        > [...] [T]he amount of time, effort, and resources spent on forecasting is not commensurate with the benefit achieved – the improvement in accuracy.

        > We spend far too many resources generating, reviewing, adjusting, and approving our forecasts, while almost invariably failing to achieve the level of accuracy desired. The evidence now shows that a large proportion of typical business forecasting efforts fail to improve the forecast, or even make it worse. So the conversation needs to change. The focus needs to change.

        > We need to shift our attention from esoteric model building to the forecasting process itself – its efficiency and its effectiveness.

        ## Linear models

        A simple linear regression can be made a global forecasting model simply by fitting in on the whole dataset.
        """
    )
    return


@app.cell
def _(freq):
    from functime.forecasting import linear_model
    from functime.preprocessing import scale, add_fourier_terms, roll
    from functime.seasonality import add_calendar_effects

    target_transforms = scale()
    feature_transforms = roll(
        window_sizes=(6, 12), stats=("mean", "std"), freq=freq
    )

    forecaster_linear = linear_model(
        freq=freq,
        lags=12,
        target_transform=target_transforms,
        feature_transform=feature_transforms,
    )
    return (
        add_calendar_effects,
        add_fourier_terms,
        feature_transforms,
        forecaster_linear,
        linear_model,
        roll,
        scale,
        target_transforms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this case, we scale the target variable and perform a couple of other common steps in feature engineering for time series:

        1. Add 12 lagged values of the target variable.
        2. Compute the rolling mean and standard deviation with window size of 6 and 12 (4 total).

        To improve our model evaluation, we also perform time-series cross validation with 5 splits of length 60 (i.e., 5 years worth of data).
        """
    )
    return


@app.cell
def _(
    entity,
    forecaster_linear,
    forecasting_horizon,
    k,
    plotting,
    sample,
    y_train,
):
    backtesting_opts = dict(
        y=y_train,
        window_size=60,  # 5 years of training data in each fold
        test_size=forecasting_horizon,
        step_size=1,
        n_splits=5,
    )

    y_linear_preds, y_linear_resids = forecaster_linear.backtest(
        **backtesting_opts
    )

    plotting.plot_backtests(
        y_train.filter(entity.is_in(sample)).collect(),
        y_linear_preds.filter(entity.is_in(sample)),
        title=f"Sample of {k} cross-validated linear forecasts",
        last_n=36,
    )
    return backtesting_opts, y_linear_preds, y_linear_resids


@app.cell
def _(rank_point_forecasts, y_linear_preds, y_train):
    rank_linear_point_forecast = rank_point_forecasts(
        y_true=y_train,
        y_pred=y_linear_preds,
        sort_by="mape",
        descending=True,
    )
    return rank_linear_point_forecast,


@app.cell
def _(
    entity,
    k,
    plotting,
    rank_linear_point_forecast,
    y_linear_preds,
    y_train,
):
    worst_linear_point_forecast = (
        rank_linear_point_forecast.select(entity).to_series().head(k)
    )

    plotting.plot_forecasts(
        y_train.filter(entity.is_in(worst_linear_point_forecast)).collect(),
        y_linear_preds.filter(entity.is_in(worst_linear_point_forecast)),
        title=f"Worst {k} linear forecasts by mean absolute percentage error (MAPE)",
    )
    return worst_linear_point_forecast,


@app.cell
def _(
    entity,
    k,
    plotting,
    rank_linear_point_forecast,
    y_linear_preds,
    y_train,
):
    best_linear_point_forecast = (
        rank_linear_point_forecast.select(entity).to_series().tail(k)
    )

    plotting.plot_forecasts(
        y_train.filter(entity.is_in(best_linear_point_forecast)).collect(),
        y_linear_preds.filter(entity.is_in(best_linear_point_forecast)),
        title=f"Best {k} linear forecasts by mean absolute percentage error (MAPE)",
    )
    return best_linear_point_forecast,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"We can even compare residuals:")
    return


@app.cell
def _(entity, k, plotting, y_linear_resids):
    from functime.evaluation import rank_residuals

    rank_linear_residuals = rank_residuals(
        y_resids=y_linear_resids, sort_by="abs_bias", descending=True
    )

    best_linear_residuals = (
        rank_linear_residuals.select(entity).tail(k).to_series()
    )

    plotting.plot_residuals(
        y_resids=y_linear_resids.filter(entity.is_in(best_linear_residuals)),
        n_bins=200,
        title=f"Top {k} linear forecast residuals",
    )
    return best_linear_residuals, rank_linear_residuals, rank_residuals


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"## Gradient Boosted Trees")
    return


@app.cell
def _(feature_transforms, target_transforms):
    from functime.forecasting import lightgbm

    forecaster_gbm = lightgbm(
        freq="1mo",
        lags=12,
        target_transform=target_transforms,
        feature_transform=feature_transforms,
    )
    return forecaster_gbm, lightgbm


@app.cell
def _(backtesting_opts, forecaster_gbm):
    y_lgb_preds, y_lgb_resids = forecaster_gbm.backtest(**backtesting_opts)
    return y_lgb_preds, y_lgb_resids


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"### Diagnostics")
    return


@app.cell
def _(entity, k, plotting, sample, y_lgb_preds, y_train):
    plotting.plot_backtests(
        y_train.filter(entity.is_in(sample)).collect(),
        y_lgb_preds.filter(entity.is_in(sample)),
        title=f"Sample of {k} cross-validated LightGBM forecasts",
        last_n=36,
    )
    return


@app.cell
def _(rank_point_forecasts, y_lgb_preds, y_train):
    rank_lgb_point_forecast = rank_point_forecasts(
        y_true=y_train,
        y_pred=y_lgb_preds,
        sort_by="mape",
        descending=True,
    )
    return rank_lgb_point_forecast,


@app.cell
def _(
    entity,
    k,
    plotting,
    rank_linear_residuals,
    rank_residuals,
    y_lgb_resids,
):
    rank_lgb_residuals = rank_residuals(
        y_resids=y_lgb_resids, sort_by="abs_bias", descending=True
    )

    best_lgb_residuals = rank_linear_residuals.select(entity).tail(k).to_series()

    plotting.plot_residuals(
        y_resids=y_lgb_resids.filter(entity.is_in(best_lgb_residuals)),
        n_bins=200,
        title=f"Top {k} LightGBM forecast residuals",
    )
    return best_lgb_residuals, rank_lgb_residuals


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # More visualisations

        Currently naive forecasters raise an error when `.backtest` is called on them. The error is caused by computing the residuals; it will be fixed before the next minor release.
        """
    )
    return


@app.cell
def _(backtesting_opts, forecaster_naive):
    from functime.backtesting import backtest
    from functime.cross_validation import sliding_window_split

    # currently raises an error; will be fixed in time for 0.10.0
    # forecaster_naive.backtest(
    #     y=y_train,
    #     test_size=forecasting_horizon,
    #     step_size=1,
    #     n_splits=5,
    #     strategy="rolling",
    #     window_size=36,
    #     conformalize=False,
    # )

    # make a new forecast each month
    step_size = 1

    # backtest: refit one year
    n_splits = 12

    cv_sliding = sliding_window_split(
        test_size=backtesting_opts["test_size"],
        step_size=backtesting_opts["step_size"],
        n_splits=backtesting_opts["n_splits"],
        window_size=backtesting_opts["window_size"],
    )

    y_bench_preds = backtest(
        forecaster=forecaster_naive,
        fh=backtesting_opts["test_size"],
        y=backtesting_opts["y"],
        cv=cv_sliding,
        residualize=False,  # currently unsupported, will be fixed in time for 0.10.0
    )
    return (
        backtest,
        cv_sliding,
        n_splits,
        sliding_window_split,
        step_size,
        y_bench_preds,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The comet plot uses sMAPE (currently, will be fixed):

        $\text{sMAPE} = \text{mean}\left(200|y_{t} - \hat{y}_{t}|/(y_{t}+\hat{y}_{t})\right)$

        This metric has its weaknesses, despite being used (e.g. in M3):

        > However, if $y_t$ is close to zero, $\hat{y}_{t}$ is also likely to be close to zero. Thus, the measure still involves division by a number close to zero, making the calculation unstable. Also, the value of sMAPE can be negative, so it is not really a measure of “absolute percentage errors” at all. [ref](https://otexts.com/fpp3/accuracy.html#percentage-errors)
        """
    )
    return


@app.cell
def _(plotting, y_bench_preds, y_lgb_preds, y_train):
    plotting.plot_fva(
        y_train,
        y_lgb_preds,
        y_bench_preds,
        title="SMAPE: Seasonal Naive (x axis) and LightGBM (y axis)",
    )
    return


@app.cell
def _(plotting, y_bench_preds, y_linear_preds, y_train):
    plotting.plot_fva(
        y_train,
        y_linear_preds,
        y_bench_preds,
        title="SMAPE: Seasonal Naive (x axis) and Linear (y axis)",
    )
    return


@app.cell
def _(plotting, y_lgb_preds, y_train):
    plotting.plot_comet(
        y_train=y_train.collect(),
        y_test=y_train.collect(),
        y_pred=y_lgb_preds,
        title="LightGBM: Coefficient of Variation (x) vs SMAPE (y)",
    )
    return


@app.cell
def _(plotting, y_linear_preds, y_train):
    plotting.plot_comet(
        y_train=y_train.collect(),
        y_test=y_train.collect(),
        y_pred=y_linear_preds,
        title="Linear Model: Coefficient of Variation (x) vs SMAPE (y)",
    )
    return


if __name__ == "__main__":
    app.run()
