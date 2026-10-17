import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize



# read relevant input files
orders_df = pd.read_csv("Data/QF_Facility_Orders.csv")
order_lines_df = pd.read_csv("Data/QF_Facility_Order_Lines.csv")


# -------- Part c --------
# get lines and units for each order
basket_df = (
    order_lines_df.groupby("order_id")
    .agg(lines=("sku_id", "size"), units=("quantity", "sum"))
    .reset_index()
)

basket_df = basket_df.merge(
    orders_df[["order_id", "day", "day_of_week", "basket_size_units"]],
    on="order_id",
    how="left"
)

# check if units match basket size in orders file
print("units match orders file:",
      (basket_df["units"] == basket_df["basket_size_units"]).mean())

# add month of order
basket_df = basket_df.merge(
    orders_df[["order_id", "placement_date"]],
    on="order_id",
    how="left"
)

basket_df["month"] = pd.to_datetime(basket_df['placement_date']).dt.month

# summary statistics for lines and units per order
basket_summary = []

for col in["lines", "units"]:
    x = basket_df[col]

    row = {
        "measure" : col + "_per_order",
        "mean" : x.mean(),
        "variance" : x.var(),
        "variance_to_mean" : x.var() / x.mean(),
        "median" : x.median(),
        "p90" : x.quantile(0.90),
        "p90" : x.quantile(0.95),
        "p99" : x.quantile(0.99),
        "max" : x.max()    
     }

    basket_summary.append(row)

basket_summary_df = pd.DataFrame(basket_summary)

print(basket_summary_df)
print("Units per line:", basket_df["units"].sum() / basket_df["lines"].sum())

# fit distribution to basket size 
basket_fits = []
fitted_pmfs = {}

for col in ["lines", "units"]:

    x = basket_df[col].values
    y = x - 1
    n = len(y)

    sizes = np.arange(1, x.max() + 1)
    observed = np.bincount(x, minlength=x.max() + 1)[1:]

    #1 + Poisson
    lam = y.mean()
    poisson_pmf = stats.poisson.pmf(sizes - 1, lam)
    poisson_loglik = stats.poisson.logpmf(y, lam).sum()

    basket_fits.append({
        "measure": col,
        "distribution": "1 + Poisson",
        "params": f"lambda={lam:.3f}",
        "num_params": 1,
        "loglik": poisson_loglik,
        "pmf": poisson_pmf
    })

    #1 + Geometric
    p_geom = 1 / (1 + y.mean())
    geom_pmf = stats.geom.pmf(sizes, p_geom)
    geom_loglik = stats.geom.logpmf(y + 1, p_geom).sum()

    basket_fits.append({
        "measure": col,
        "distribution": "1 + Geometric",
        "params": f"p={p_geom:.3f}",
        "num_params": 1,
        "loglik": geom_loglik,
        "pmf": geom_pmf
    })

    #1 + Negative binomial
    mean_y = y.mean()
    var_y = y.var()
    r_start = mean_y**2 / (var_y - mean_y)

    nb_result = minimize(
        lambda theta: -stats.nbinom.logpmf(
            y, np.exp(theta[0]), 1 / (1 + np.exp(-theta[1]))
        ).sum(),
        x0=[np.log(r_start), np.log(r_start / mean_y)],
        method="Nelder-Mead"
    )

    r_nb = np.exp(nb_result.x[0])
    p_nb = 1 / (1 + np.exp(-nb_result.x[1]))
    nb_pmf = stats.nbinom.pmf(sizes - 1, r_nb, p_nb)

    basket_fits.append({
        "measure": col,
        "distribution": "1 + Negative binomial",
        "params": f"r={r_nb:.3f}, p={p_nb:.3f}",
        "num_params": 2,
        "loglik": -nb_result.fun,
        "pmf": nb_pmf
    })

    # goodness of fit for each distribution
    for fit in basket_fits:
        if fit["measure"] != col:
            continue

        expected = fit["pmf"] * n
        expected[-1] += n - expected.sum()

        # pool the upper tail so that every bin has expected count >=5
        cut = len(expected) - 1 - np.argmax(np.cumsum(expected[::-1]) >= 5)
        obs_binned = np.append(observed[:cut], observed[cut:].sum())
        exp_binned = np.append(expected[:cut], expected[cut:].sum())

        fit["AIC"] = 2 * fit["num_params"] - 2 * fit["loglik"]
        fit["chi2"] = ((obs_binned - exp_binned)**2 / exp_binned).sum()
        fit["chi2_dof"] = len(obs_binned) - 1 - fit["num_params"]
        fit["max_pmf_error_pct"] = np.abs(observed / n - fit["pmf"]).max() * 100

        fitted_pmfs[(col, fit["distribution"])] = fit["pmf"]


basket_fits_df = (
    pd.DataFrame(basket_fits)
    .drop(columns ="pmf")
    .sort_values(["measure", "AIC"])
    .reset_index(drop=True)
)

print(basket_fits_df[["measure", "distribution", "params", "AIC",
                      "chi2", "chi2_dof", "max_pmf_error_pct"]])


#basket size by weekday and by month
basket_by_weekday_df = (
    basket_df.groupby("day_of_week")
    .agg(orders=("order_id", "size"),
         mean_lines=("lines", "mean"),
         mean_units=("units", "mean"))
    .reset_index()
)

basket_by_month_df = (
    basket_df.groupby("month")
    .agg(orders=("order_id", "size"),
         mean_lines=("lines", "mean"),
         mean_units=("units", "mean"))
    .reset_index()
)

print(basket_by_weekday_df)
print(basket_by_month_df)

#test if basket size differs by weekday / month
weekday_test = stats.kruskal(
    *[group["lines"].values for _, group in basket_df.groupby("day_of_week")]
)
month_test = stats.kruskal(
    *[group["lines"].values for _, group in basket_df.groupby("month")]
)

print("Kruskal-Wallis weekday p-value:", weekday_test.pvalue)
print("Weekday range of mean lines:",
      basket_by_weekday_df["mean_lines"].max() / basket_by_weekday_df["mean_lines"].min() - 1)
print("Kruskal-Wallis month p-value:", month_test.pvalue)
print("Month range of mean lines:",
      basket_by_month_df["mean_lines"].max() / basket_by_month_df["mean_lines"].min() - 1)


# empirical basket size distribution
max_size = max(basket_df["lines"].max(), basket_df["units"].max())

basket_distribution_df = pd.DataFrame({"size": np.arange(1, max_size + 1)})

for col in ["lines", "units"]:
    counts = (
        basket_df[col].value_counts()
        .reindex(basket_distribution_df["size"], fill_value=0)
        .values
    )


    basket_distribution_df[col + "_orders"] = counts
    basket_distribution_df[col + "_prob"] = counts / counts.sum()
    basket_distribution_df[col + "_cum_prob"] = np.cumsum(counts) / counts.sum()

print(basket_distribution_df.head(15))

# plot observed vs best fit, and cumulative distribution
best_fit = basket_fits_df[basket_fits_df["measure"] == "lines"].iloc[0]["distribution"]
best_pmf = fitted_pmfs[("lines", best_fit)]

sizes = basket_distribution_df["size"].values
show = sizes <= 15

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(sizes[show], basket_distribution_df["lines_prob"][show] * 100, 
            label="Observed")
axes[0].plot(sizes[show], best_pmf[:show.sum()] * 100, "o-", color="tab:orange",
             label="Fitted: " + best_fit)
axes[0].set_xlabel("Order lines per order")
axes[0].set_ylabel("% of orders")
axes[0].set_title("Basket size distribution")
axes[0].legend()

axes[1].plot(sizes[show], basket_distribution_df["lines_cum_prob"][show] * 100,
             "o-", label="Lines")
axes[1].plot(sizes[show], basket_distribution_df["units_cum_prob"][show] * 100,
             "s-", label="Units")
axes[1].axhline(90, color="grey", linewidth=0.8)
axes[1].axhline(99, color="grey", linewidth= 0.8)
axes[1].set_xlabel("Basket size")
axes[1].set_ylabel("Cumulative % of orders")
axes[1].set_title("Cumulative basket size distribution")
axes[1].legend()

plt.tight_layout()
plt.show()