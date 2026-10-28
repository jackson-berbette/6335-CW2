import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson, nbinom
from sklearn.metrics import mean_absolute_error, mean_squared_error
from IPython.display import display



# read relevant input files
orders_df = pd.read_csv("Data/QF_Facility_Orders.csv")
order_lines_df = pd.read_csv("Data/QF_Facility_Order_Lines.csv")


# -------- Part a --------
# add day of order to order lines
order_lines_df = order_lines_df.merge(
    orders_df[["order_id", "day", "day_of_week"]],
    on="order_id",
    how="left"
)

daily_demand = (
    order_lines_df.groupby(["sku_id", "day", "day_of_week"])["quantity"]
    .sum().reset_index(name="demand")
)

day_weekday = (
    orders_df[["day", "day_of_week"]]
    .drop_duplicates()
    .set_index("day")["day_of_week"]
)

# fill all missing sku-day combos with zero demand
all_skus = sorted(order_lines_df["sku_id"].unique())
all_days = sorted(order_lines_df["day"].unique())

full_index = pd.MultiIndex.from_product(
    [all_skus, all_days], names=["sku_id", "day"]
)

daily_demand = (
    daily_demand.drop(columns="day_of_week")
    .set_index(["sku_id", "day"])
    .reindex(full_index, fill_value=0).reset_index()
)

daily_demand["day_of_week"] = daily_demand["day"].map(day_weekday)

# fit negative binomial model
final_models = {}
final_predictions = []
final_failed_skus = {}

for sku_id, sku_data in daily_demand.groupby("sku_id"):

    df = sku_data.copy().sort_values("day")
    df["time"] = (df["day"] - 183) / 365

    # Handle products with no demand during the year
    if df["demand"].sum() == 0:
        df["mean_estimate"] = 0.0
        df["alpha"] = np.nan
        df["variance_estimate"] = 0.0
        df["nb_size"] = np.nan
        df["nb_prob"] = 1.0

        final_predictions.append(df)
        final_models[sku_id] = None
        continue

    try:
        # Fit negative binomial using the entire historical year
        model = smf.negativebinomial(
            formula="demand ~ time + C(day_of_week)",
            data=df
        ).fit(disp=0, maxiter=500)

        # Estimated daily mean
        df["mean_estimate"] = model.predict(df)

        # Estimated SKU-specific dispersion
        alpha = float(model.params["alpha"])

        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid dispersion estimate: {alpha}")

        df["alpha"] = alpha

        # Negative binomial variance
        df["variance_estimate"] = (
            df["mean_estimate"]
            + alpha * df["mean_estimate"]**2
        )

        # Parameters for scipy.stats.nbinom
        # Variance = mean + alpha * mean^2
        df["nb_size"] = 1 / alpha
        df["nb_prob"] = (
            df["nb_size"]
            / (df["nb_size"] + df["mean_estimate"])
        )

        final_models[sku_id] = model
        final_predictions.append(df)

    except Exception as e:
        print(f"{sku_id}: NB failed ({e}), using Poisson")

        model = smf.poisson(
            formula="demand ~ time + C(day_of_week)",
            data=df
        ).fit(disp=0, maxiter=500)

        if not model.mle_retvals.get("converged", False):
            raise ValueError(f"Poisson also failed for {sku_id}")

        alpha = 0.0

# Combine all product-day estimates
if final_predictions:
    final_demand_distributions = pd.concat(
        final_predictions,
        ignore_index=True
    )

print("Products:", daily_demand["sku_id"].nunique())
print("Successfully processed:", len(final_models))
print("Failed fits:", len(final_failed_skus))
print("Product-day rows:", len(final_demand_distributions))

# get dataframe for product model parameters
model_parameters = []

for sku_id, model in final_models.items():
    if model is None:
        continue

    params = model.params

    row = {
        "sku_id": sku_id,
        "intercept": params.get("Intercept", np.nan),
        "time_trend": params.get("time", np.nan),
        "dispersion_alpha": params.get("alpha", np.nan)
    }

    for name, value in params.items():
        if name.startswith("C(day_of_week)"):
            row[name] = value

    model_parameters.append(row)

model_parameters_df = pd.DataFrame(model_parameters)

print(model_parameters_df)


# -------- Export future daily demand distributions --------

# Create requested forecast dates (inclusive)
forecast_dates = pd.date_range(
    start="2027-09-07",
    end="2027-12-27",
    freq="D"
)

# Create every SKU-date combination
forecast_index = pd.MultiIndex.from_product(
    [all_skus, forecast_dates],
    names=["sku_id", "date"]
)

forecast_df = forecast_index.to_frame(index=False)

# Match the day-of-week encoding used in Task 1
# Historical encoding: 0 = Monday, ..., 6 = Sunday
forecast_df["day_of_week"] = forecast_df["date"].dt.dayofweek

# IMPORTANT:
# Historical day 1 must correspond to a known calendar date.
# Replace this with the actual date of historical day 1.
historical_start_date = pd.Timestamp("2026-01-01")

# Reconstruct the same time variable used in Task 1
forecast_df["day"] = (
    forecast_df["date"] - historical_start_date
).dt.days + 1

forecast_df["time"] = (forecast_df["day"] - 183) / 365

# Predict each SKU's future demand distribution
forecast_results = []

for sku_id, sku_forecast in forecast_df.groupby("sku_id"):

    df = sku_forecast.copy()
    model = final_models.get(sku_id)

    if sku_id not in final_models:
        raise ValueError(f"No fitted model for {sku_id}")

    if model is None:
        # SKU had zero historical demand
        df["mean_units"] = 0.0
        df["sd_units"] = 0.0

    else:
        # Predicted daily mean
        df["mean_units"] = model.predict(df)

        # Negative Binomial dispersion
        alpha = float(model.params["alpha"])

        # Predicted daily standard deviation
        df["sd_units"] = np.sqrt(
            df["mean_units"] +
            alpha * df["mean_units"]**2
        )

    forecast_results.append(
        df[["sku_id", "date", "mean_units", "sd_units"]]
    )

# Combine results
demand_forecast = pd.concat(
    forecast_results,
    ignore_index=True
)

# Format date for CSV
demand_forecast["date"] = (
    demand_forecast["date"].dt.strftime("%Y-%m-%d")
)

# Validate output
expected_rows = len(all_skus) * len(forecast_dates)

assert len(all_skus) == 2000
assert len(demand_forecast) == expected_rows
assert not demand_forecast.isna().any().any()
assert (demand_forecast[["mean_units", "sd_units"]] >= 0).all().all()

# Save CSV
demand_forecast.to_csv(
    "Output/Task1_QF_Daily_Demand_Forecast.csv",
    index=False
)

print("Total rows:", len(demand_forecast))
print("Unique SKUs:", demand_forecast["sku_id"].nunique())
print("Unique dates:", demand_forecast["date"].nunique())

display(demand_forecast.head(10))
