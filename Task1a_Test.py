import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson, nbinom
from sklearn.metrics import mean_absolute_error, mean_squared_error
from IPython.display import display


# This file tests to find the most appropriate model. It first tries Poisson.
# It compares to Negative Binomial. It is decided that we will fit
# a negative binomial model for all products.

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

# fit trend and seasonality model (Poisson)
def fit_sku_model(sku_data):
    df = sku_data.copy().sort_values("day")

    # Center time to make the trend easier to interpret
    df["time"] = (df["day"] - 183) / 365

    if df["demand"].sum() == 0:
        return None, df.assign(mean_estimate=0.0)

    model = smf.glm(
        formula="demand ~ time + C(day_of_week)",
        data=df,
        family=sm.families.Poisson()
    ).fit()

    df["mean_estimate"] = model.predict(df)

    return model, df

sku_models = {}
all_predictions = []
failed_skus = {}

for sku_id, sku_data in daily_demand.groupby("sku_id"):
    try:
        model, predictions = fit_sku_model(sku_data)

        sku_models[sku_id] = model
        all_predictions.append(predictions)

    except Exception as e:
        failed_skus[sku_id] = str(e)

if all_predictions:
    predictions_df = pd.concat(all_predictions, ignore_index=True)

# test for overdispersion
dispersion_results = []

for sku_id, model in sku_models.items():
    if model is None:
        continue

    dispersion_results.append({
        "sku_id": sku_id,
        "dispersion": model.pearson_chi2 / model.df_resid,
        "mean_demand": predictions_df.loc[
            predictions_df["sku_id"] == sku_id, "demand"
        ].mean()
    })

dispersion_df = pd.DataFrame(dispersion_results)

print(dispersion_df["dispersion"].describe())

def classify_dispersion(phi):
    if pd.isna(phi):
        return "Undefined"

    if phi < 0.8:
        return "Possible underdispersion"
    elif phi <= 1.2:
        return "Near Poisson"
    else:
        return "Overdispersion"


dispersion_df["category"] = (
    dispersion_df["dispersion"].apply(classify_dispersion)
)

print(dispersion_df["category"].value_counts())

print(
    dispersion_df.groupby("category")["dispersion"]
    .agg(["count", "mean", "median", "max"])
)

overdispersed_skus = dispersion_df.loc[
    dispersion_df["category"] == "Overdispersion",
    "sku_id"
].tolist()

# test negative binomial model for overdispersed products
def fit_sku_nb_model(sku_data):
    df = sku_data.copy().sort_values("day")

    # Center time to make the trend easier to interpret
    df["time"] = (df["day"] - 183) / 365

    # Handle products with zero demand throughout the year
    if df["demand"].sum() == 0:
        return None, df.assign(mean_estimate=0.0)

    # Fit negative binomial regression
    model = smf.negativebinomial(
        formula="demand ~ time + C(day_of_week)",
        data=df
    ).fit(disp=0)

    # Estimate mean demand for each day
    df["mean_estimate"] = model.predict(df)

    return model, df

nb_models = {}
nb_predictions = []
nb_failed_skus = {}

for sku_id in overdispersed_skus:
    sku_data = daily_demand[
        daily_demand["sku_id"] == sku_id
    ]

    try:
        model, predictions = fit_sku_nb_model(sku_data)

        nb_models[sku_id] = model
        nb_predictions.append(predictions)

    except Exception as e:
        nb_failed_skus[sku_id] = str(e)

if nb_predictions:
    nb_predictions_df = pd.concat(
        nb_predictions, ignore_index=True
    )

print("Negative binomial models fitted:", len(nb_models))
print("Failed fits:", len(nb_failed_skus))

# compare models
comparison_results = []
comparison_predictions = []
failed_comparisons = {}

for sku_id in overdispersed_skus:

    df = daily_demand[
        daily_demand["sku_id"] == sku_id
    ].copy().sort_values("day")

    # Chronological train/test split
    train = df[df["day"] <= 300].copy()
    test = df[df["day"] > 300].copy()

    # Use the same time definition for both models
    train["time"] = (train["day"] - 183) / 365
    test["time"] = (test["day"] - 183) / 365

    # Skip products with no training demand
    if train["demand"].sum() == 0:
        failed_comparisons[sku_id] = "Zero training demand"
        continue

    try:
        # -------- Poisson model --------
        poisson_model = smf.glm(
            formula="demand ~ time + C(day_of_week)",
            data=train,
            family=sm.families.Poisson()
        ).fit()

        poisson_mean = np.maximum(
            poisson_model.predict(test), 1e-10
        )

        # -------- Negative binomial model --------
        nb_model = smf.negativebinomial(
            formula="demand ~ time + C(day_of_week)",
            data=train
        ).fit(disp=0, maxiter=200)

        nb_mean = np.maximum(
            nb_model.predict(test), 1e-10
        )

        # Estimated negative binomial dispersion parameter
        alpha = float(nb_model.params["alpha"])
        alpha = max(alpha, 1e-10)

        # NB parameterization:
        # variance = mean + alpha * mean^2
        size = 1.0 / alpha
        nb_prob = size / (size + nb_mean)

        actual = test["demand"].to_numpy()

        # -------- Mean prediction metrics --------
        for model_name, pred in [
            ("Poisson", poisson_mean),
            ("Negative Binomial", nb_mean)
        ]:
            comparison_results.append({
                "sku_id": sku_id,
                "model": model_name,
                "MAE": mean_absolute_error(actual, pred),
                "RMSE": np.sqrt(
                    mean_squared_error(actual, pred)
                ),
                # Positive bias means underforecasting
                "Bias": np.mean(actual - pred)
            })

        # -------- Held-out distribution metrics --------
        poisson_nll = -np.mean(
            poisson.logpmf(actual, poisson_mean)
        )

        nb_nll = -np.mean(
            nbinom.logpmf(actual, size, nb_prob)
        )

        comparison_results[-2]["NLL"] = poisson_nll
        comparison_results[-1]["NLL"] = nb_nll

        # Save predictions for later analysis
        comparison_predictions.append(pd.DataFrame({
            "sku_id": sku_id,
            "day": test["day"].to_numpy(),
            "actual": actual,
            "poisson_mean": poisson_mean,
            "nb_mean": nb_mean
        }))

    except Exception as e:
        failed_comparisons[sku_id] = str(e)

# Combine results
results_df = pd.DataFrame(comparison_results)

if comparison_predictions:
    test_predictions_df = pd.concat(
        comparison_predictions, ignore_index=True
    )

print("Successfully evaluated SKUs:",
      results_df["sku_id"].nunique() if not results_df.empty else 0)
print("Failed comparisons:", len(failed_comparisons))

# results
metric_summary = (
    results_df.groupby("model")[["MAE", "RMSE", "Bias", "NLL"]]
    .mean()
    .round(4)
)

print(metric_summary)


nll_comparison = results_df.pivot(
    index="sku_id",
    columns="model",
    values="NLL"
).dropna()

nll_comparison["winner"] = np.where(
    nll_comparison["Negative Binomial"]
    < nll_comparison["Poisson"],
    "Negative Binomial",
    "Poisson"
)

print(nll_comparison["winner"].value_counts())

nll_comparison["nll_improvement"] = (
    nll_comparison["Poisson"]
    - nll_comparison["Negative Binomial"]
)

print(nll_comparison["nll_improvement"].describe())

# test negative binomial for near-poisson skus
near_poisson_skus = dispersion_df.loc[
    dispersion_df["category"] == "Near Poisson",
    "sku_id"
].tolist()

print("Near-Poisson products:", len(near_poisson_skus))

near_comparison_results = []
near_comparison_predictions = []
near_failed_skus = {}

for sku_id in near_poisson_skus:

    df = daily_demand[
        daily_demand["sku_id"] == sku_id
    ].copy().sort_values("day")

    train = df[df["day"] <= 300].copy()
    test = df[df["day"] > 300].copy()

    train["time"] = (train["day"] - 183) / 365
    test["time"] = (test["day"] - 183) / 365

    if train["demand"].sum() == 0:
        near_failed_skus[sku_id] = "Zero training demand"
        continue

    try:
        # Poisson model
        poisson_model = smf.glm(
            formula="demand ~ time + C(day_of_week)",
            data=train,
            family=sm.families.Poisson()
        ).fit()

        poisson_mean = np.maximum(
            poisson_model.predict(test), 1e-10
        )

        # Negative binomial model
        nb_model = smf.negativebinomial(
            formula="demand ~ time + C(day_of_week)",
            data=train
        ).fit(disp=0, maxiter=200)

        nb_mean = np.maximum(
            nb_model.predict(test), 1e-10
        )

        actual = test["demand"].to_numpy()

        # Calculate mean prediction metrics
        for model_name, pred in [
            ("Poisson", poisson_mean),
            ("Negative Binomial", nb_mean)
        ]:
            near_comparison_results.append({
                "sku_id": sku_id,
                "model": model_name,
                "MAE": mean_absolute_error(actual, pred),
                "RMSE": np.sqrt(
                    mean_squared_error(actual, pred)
                ),
                "Bias": np.mean(actual - pred)
            })

        # Calculate held-out negative log-likelihood
        poisson_nll = -np.mean(
            poisson.logpmf(actual, poisson_mean)
        )

        alpha = max(float(nb_model.params["alpha"]), 1e-10)
        size = 1.0 / alpha
        nb_prob = size / (size + nb_mean)

        nb_nll = -np.mean(
            nbinom.logpmf(actual, size, nb_prob)
        )

        near_comparison_results[-2]["NLL"] = poisson_nll
        near_comparison_results[-1]["NLL"] = nb_nll

        near_comparison_predictions.append(pd.DataFrame({
            "sku_id": sku_id,
            "day": test["day"].to_numpy(),
            "actual": actual,
            "poisson_mean": poisson_mean,
            "nb_mean": nb_mean
        }))

    except Exception as e:
        near_failed_skus[sku_id] = str(e)

# Combine results
near_results_df = pd.DataFrame(near_comparison_results)

if near_comparison_predictions:
    near_test_predictions_df = pd.concat(
        near_comparison_predictions, ignore_index=True
    )

print("Evaluated SKUs:", near_results_df["sku_id"].nunique())
print("Failed comparisons:", len(near_failed_skus))

print(
    near_results_df.groupby("model")[["MAE", "RMSE", "Bias", "NLL"]]
    .mean()
    .round(4)
)

near_nll_comparison = near_results_df.pivot(
    index="sku_id",
    columns="model",
    values="NLL"
).dropna()

near_nll_comparison["nll_improvement"] = (
    near_nll_comparison["Poisson"]
    - near_nll_comparison["Negative Binomial"]
)

near_nll_comparison["winner"] = np.where(
    near_nll_comparison["nll_improvement"] > 0,
    "Negative Binomial",
    "Poisson"
)

print(near_nll_comparison["winner"].value_counts())
print(near_nll_comparison["nll_improvement"].describe())

nb_models = {}
nb_predictions = []
nb_failed_skus = {}

for sku_id, sku_data in daily_demand.groupby("sku_id"):
    try:
        model, predictions = fit_sku_nb_model(sku_data)

        nb_models[sku_id] = model
        nb_predictions.append(predictions)

    except Exception as e:
        nb_failed_skus[sku_id] = str(e)

if nb_predictions:
    predictions_df = pd.concat(nb_predictions, ignore_index=True)

# investigate time-varying mean
#Choose a SKU to inspect
sku_id = "P0001"  # Replace with a SKU you want to investigate

sku_data = (
    daily_demand[daily_demand["sku_id"] == sku_id]
    .copy()
    .sort_values("day")
)

sku_data["time"] = (sku_data["day"] - 183) / 365

# Fit the model on the first 300 days
train = sku_data[sku_data["day"] <= 300]
test = sku_data[sku_data["day"] > 300]

model = smf.negativebinomial(
    formula="demand ~ time + C(day_of_week)",
    data=train
).fit(disp=0)

# Predict demand for both periods
sku_data["predicted_mean"] = np.nan
sku_data.loc[train.index, "predicted_mean"] = model.predict(train)
sku_data.loc[test.index, "predicted_mean"] = model.predict(test)

# Plot actual demand and predicted mean
plt.figure(figsize=(12, 5))
plt.plot(
    sku_data["day"], sku_data["demand"],
    label="Actual demand", alpha=0.6
)
plt.plot(
    sku_data["day"], sku_data["predicted_mean"],
    label="Predicted mean", linewidth=2
)
plt.axvline(300, linestyle="--", color="gray", label="Train/test split")
plt.xlabel("Day of year")
plt.ylabel("Daily demand")
plt.title(f"Actual vs. predicted demand: {sku_id}")
plt.legend()
plt.tight_layout()
plt.show()

# Add a quadratic time term
train["time_squared"] = train["time"] ** 2
test["time_squared"] = test["time"] ** 2

# Fit a more flexible trend model
quadratic_model = smf.negativebinomial(
    formula="demand ~ time + time_squared + C(day_of_week)",
    data=train
).fit(disp=0)

# Predict the held-out period
test = test.copy()
test["linear_prediction"] = model.predict(test)
test["quadratic_prediction"] = quadratic_model.predict(test)

# Compare the predictions with actual demand
plt.figure(figsize=(12, 5))
plt.plot(test["day"], test["demand"], label="Actual demand")
plt.plot(test["day"], test["linear_prediction"], label="Linear trend")
plt.plot(test["day"], test["quadratic_prediction"], label="Quadratic trend")
plt.xlabel("Day of year")
plt.ylabel("Daily demand")
plt.title("Linear vs. quadratic trend: P0001")
plt.legend()
plt.tight_layout()
plt.show()


# test for time-varying variability
diagnostic_rows = []
failed_skus = {}

for sku_id, sku_data in daily_demand.groupby("sku_id"):
    sku_data = sku_data.copy().sort_values("day")
    sku_data["time"] = (sku_data["day"] - 183) / 365

    train = sku_data[sku_data["day"] <= 300]
    test = sku_data[sku_data["day"] > 300].copy()

    # Skip SKUs with no training demand
    if train["demand"].sum() == 0:
        continue

    try:
        model = smf.negativebinomial(
            formula="demand ~ time + C(day_of_week)",
            data=train
        ).fit(disp=0)

        alpha = float(model.params["alpha"])

        if not np.isfinite(alpha) or alpha <= 0:
            failed_skus[sku_id] = f"Invalid alpha: {alpha}"
            continue

        # Predict mean and variance on unseen days
        test["predicted_mean"] = model.predict(test)
        test["predicted_variance"] = (
            test["predicted_mean"]
            + alpha * test["predicted_mean"]**2
        )

        # Standardized residuals
        test["std_residual"] = (
            test["demand"] - test["predicted_mean"]
        ) / np.sqrt(test["predicted_variance"].clip(lower=1e-8))

        test["sku_id"] = sku_id
        diagnostic_rows.append(
            test[[
                "sku_id", "day", "demand",
                "predicted_mean", "predicted_variance",
                "std_residual"
            ]]
        )

    except Exception as e:
        failed_skus[sku_id] = str(e)

test_diagnostics = pd.concat(diagnostic_rows, ignore_index=True)

print("SKUs evaluated:", test_diagnostics["sku_id"].nunique())
print("SKUs skipped or failed:", len(failed_skus))

test_diagnostics["period"] = pd.cut(
    test_diagnostics["day"],
    bins=[300, 322, 343, 365],
    labels=["Days 301-322", "Days 323-343", "Days 344-365"]
)

variability_summary = (
    test_diagnostics.groupby("period", observed=True)
    .agg(
        mean_residual=("std_residual", "mean"),
        mean_squared_residual=(
            "std_residual",
            lambda x: np.mean(x**2)
        ),
        observations=("std_residual", "size")
    )
    .reset_index()
)

display(variability_summary)

p0001 = test_diagnostics[
    test_diagnostics["sku_id"] == "P0001"
].copy()

plt.figure(figsize=(12, 4))
plt.plot(p0001["day"], p0001["std_residual"], marker=".", linewidth=1)
plt.axhline(0, linestyle="--", color="gray")
plt.axhline(2, linestyle=":", color="red")
plt.axhline(-2, linestyle=":", color="red")
plt.xlabel("Day of year")
plt.ylabel("Standardized residual")
plt.title("Negative binomial residuals: P0001")
plt.tight_layout()
plt.show()
