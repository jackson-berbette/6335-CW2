import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import chi2, chi2_contingency



# read relevant input files
orders_df = pd.read_csv("Data/QF_Facility_Orders.csv")
design_params_df = pd.read_csv("Data/QF_Facility_Design_Parameters.csv")

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# operating window (10:00-18:00) and capacity planning interval (60 min)
def get_parameter(name):
    return float(
        design_params_df.loc[design_params_df["parameter"] == name, "value"].iloc[0]
    )

open_minute = get_parameter("order_day_start_minute")
close_minute = get_parameter("order_day_end_minute")
interval_minutes = get_parameter("planning_interval_minutes")


# -------- Part a: day-of-week --------
daily_orders = (
    orders_df.groupby(["day", "day_of_week", "weekend"])
    .size().reset_index(name="orders")
)

# descriptive profile by weekday
dow_summary = (
    daily_orders.groupby("day_of_week")["orders"]
    .agg(days="count", total="sum", mean="mean", std="std", min="min", max="max")
)
dow_summary["cv"] = dow_summary["std"] / dow_summary["mean"]
dow_summary["share"] = dow_summary["total"] / dow_summary["total"].sum()
dow_summary["index_vs_avg_day"] = dow_summary["mean"] / daily_orders["orders"].mean()
dow_summary.index = day_names

print("Daily orders - mean:", round(daily_orders["orders"].mean(), 1),
      "std:", round(daily_orders["orders"].std(), 1))
print(dow_summary.round(3))

# fit negative binomial with trend so weekday effects are not biased by growth
# (same structure as Task 1, at order level instead of SKU level)
daily_orders["time"] = (daily_orders["day"] - 183) / 365

dow_model = smf.negativebinomial(
    formula="orders ~ time + C(day_of_week)",
    data=daily_orders
).fit(disp=0, maxiter=200)

# likelihood-ratio test: is day-of-week significant?
trend_model = smf.negativebinomial(
    formula="orders ~ time",
    data=daily_orders
).fit(disp=0, maxiter=200)

lr_stat = 2 * (dow_model.llf - trend_model.llf)
lr_pvalue = chi2.sf(lr_stat, df=6)

# weekday multipliers relative to Monday, after removing trend
dow_effects = pd.DataFrame({
    "multiplier_vs_mon": [1.0] + [
        np.exp(dow_model.params[f"C(day_of_week)[T.{d}]"]) for d in range(1, 7)
    ],
    "p_value": [np.nan] + [
        dow_model.pvalues[f"C(day_of_week)[T.{d}]"] for d in range(1, 7)
    ]
}, index=day_names)

print("Annual growth factor:", round(np.exp(dow_model.params["time"]), 3))
print("Dispersion alpha:", round(dow_model.params["alpha"], 4))
print("LR test day-of-week: stat =", round(lr_stat, 1), "p =", lr_pvalue)
print(dow_effects.round(4))

# peak days: weekdays whose mean is above the average day
peak_days = dow_summary.index[dow_summary["index_vs_avg_day"] > 1].tolist()
print("Peak operating days:", peak_days)


# -------- Part b: time-of-day --------
# placement_minute is minutes from day 0 (day x 1440 + minute of day)
orders_df["minute_of_day"] = orders_df["placement_minute"] - orders_df["day"] * 1440

print("Minute-of-day range:", orders_df["minute_of_day"].min(),
      "-", orders_df["minute_of_day"].max())

# assign each order to a planning interval within the 10:00-18:00 window
n_intervals = int((close_minute - open_minute) / interval_minutes)
orders_df["interval"] = (
    ((orders_df["minute_of_day"] - open_minute) // interval_minutes)
    .clip(0, n_intervals - 1).astype(int)
)
interval_labels = [
    f"{int((open_minute + i * interval_minutes) // 60):02d}:00"
    for i in range(n_intervals)
]

# orders per day per interval (fill empty intervals with zero)
interval_orders = (
    orders_df.groupby(["day", "day_of_week", "interval"])
    .size().reset_index(name="orders")
)

full_index = pd.MultiIndex.from_product(
    [daily_orders["day"], range(n_intervals)], names=["day", "interval"]
)

interval_orders = (
    interval_orders.drop(columns="day_of_week")
    .set_index(["day", "interval"])
    .reindex(full_index, fill_value=0).reset_index()
)
interval_orders["day_of_week"] = interval_orders["day"].map(
    daily_orders.set_index("day")["day_of_week"]
)

# share of the day's orders in each interval
interval_orders["day_total"] = (
    interval_orders.groupby("day")["orders"].transform("sum")
)
interval_orders["share"] = interval_orders["orders"] / interval_orders["day_total"]

# overall intraday profile
intraday_profile = (
    interval_orders.groupby("interval")
    .agg(mean_orders=("orders", "mean"),
         std_orders=("orders", "std"),
         mean_share=("share", "mean"))
)
intraday_profile["index_vs_avg_interval"] = (
    intraday_profile["mean_share"] * n_intervals
)
intraday_profile.index = interval_labels

print(intraday_profile.round(3))

# share of each weekday's orders by interval (rows: weekday, cols: interval)
dow_interval_share = (
    interval_orders.groupby(["day_of_week", "interval"])["share"]
    .mean().unstack()
)
dow_interval_share.index = day_names
dow_interval_share.columns = interval_labels

print(dow_interval_share.round(3))

# chi-square test: is the intraday shape the same on every weekday?

dow_interval_counts = (
    interval_orders.groupby(["day_of_week", "interval"])["orders"]
    .sum().unstack()
)
chi2_stat, chi2_pvalue, chi2_dof, _ = chi2_contingency(dow_interval_counts)
print("Chi-square same intraday shape across weekdays: stat =",
      round(chi2_stat, 1), "dof =", chi2_dof, "p =", chi2_pvalue)

# peak periods: intervals carrying more than an even share of the day
peak_intervals = intraday_profile.index[
    intraday_profile["index_vs_avg_interval"] > 1
].tolist()
print("Peak periods within the day:", peak_intervals)


# -------- Part c: peak day x period --------
# mean orders per interval for each weekday (rows: weekday, cols: interval)
expected_interval_orders = (
    interval_orders.groupby(["day_of_week", "interval"])["orders"]
    .mean().unstack()
)
expected_interval_orders.index = day_names
expected_interval_orders.columns = interval_labels

# peak-to-average ratio: busiest interval vs average interval of that day
peak_summary = pd.DataFrame({
    "mean_daily_orders": dow_summary["mean"],
    "peak_interval": expected_interval_orders.idxmax(axis=1),
    "peak_interval_orders": expected_interval_orders.max(axis=1),
    "avg_interval_orders": expected_interval_orders.mean(axis=1),
})
peak_summary["peak_to_avg"] = (
    peak_summary["peak_interval_orders"] / peak_summary["avg_interval_orders"]
)

print(expected_interval_orders.round(1))
print(peak_summary.round(2).to_string())

# busiest single interval in the year
busiest_interval = interval_orders.loc[interval_orders["orders"].idxmax()]
print("Busiest interval: day", int(busiest_interval["day"]),
      day_names[int(busiest_interval["day_of_week"])],
      interval_labels[int(busiest_interval["interval"])],
      "orders:", int(busiest_interval["orders"]))

# 95th percentile interval load per weekday, for capacity planning
interval_p95 = (
    interval_orders.groupby(["day_of_week", "interval"])["orders"]
    .quantile(0.95).unstack()
)
interval_p95.index = day_names
interval_p95.columns = interval_labels
print(interval_p95.round(1))


# -------- Plots --------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].bar(day_names, dow_summary["mean"], yerr=dow_summary["std"], capsize=4)
axes[0].axhline(daily_orders["orders"].mean(), color="grey", linestyle="--")
axes[0].set_title("Mean daily orders by weekday")
axes[0].set_ylabel("Orders per day")

for dow in day_names:
    axes[1].plot(interval_labels, dow_interval_share.loc[dow], marker="o", label=dow)
axes[1].axhline(1 / n_intervals, color="grey", linestyle="--")
axes[1].set_title("Share of daily orders by hour")
axes[1].set_ylabel("Share of day")
axes[1].legend()

im = axes[2].imshow(expected_interval_orders, aspect="auto", cmap="viridis")
axes[2].set_xticks(range(n_intervals), interval_labels)
axes[2].set_yticks(range(7), day_names)
axes[2].set_title("Expected orders per hour")
fig.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.show()
