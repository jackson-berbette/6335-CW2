import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# read relevant input files
orders_df = pd.read_csv("Data/QF_Facility_Orders.csv")
design_params_df = pd.read_csv("Data/QF_Facility_Design_Parameters.csv")

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# required OTS range (minutes) for each scenario
ots_scenarios = {
    "Fast": (4, 5),
    "Medium": (5, 7),
    "Slow": (8, 10),
}

# targeted on-time service levels
def get_parameter(name):
    return float(
        design_params_df.loc[design_params_df["parameter"] == name, "value"].iloc[0]
    )

service_levels = {
    "minimum": get_parameter("minimum_on_time_stage_share"),
    "stretch": get_parameter("stretch_on_time_stage_share"),
}


# -------- Part f --------
# OTD = OTS + RTA + PTD
#   PTD: pickup-to-delivery, known from the order's destination
#   RTA: time a staged order waits for its courier; the courier arrives
#        courier_arrival_offset_minutes after the latest staged-ready time,
#        and an early courier (negative offset) adds no wait
orders_df["ptd"] = orders_df["estimated_delivery_minutes"]
orders_df["rta"] = orders_df["courier_arrival_offset_minutes"].clip(lower=0)

# period = week x day of week x hour of day
orders_df["week"] = orders_df["day"] // 7
orders_df["hour"] = pd.to_datetime(orders_df["placement_timestamp"]).dt.hour

print("Orders:", len(orders_df))
print("Share of couriers arriving early (no wait):",
      (orders_df["courier_arrival_offset_minutes"] <= 0).mean())
print(orders_df[["ptd", "rta"]].describe().round(2))

# check how much PTD and RTA move with period and destination
print("Corr PTD vs distance:",
      round(orders_df["ptd"].corr(orders_df["customer_distance_miles"]), 4))
print("Corr RTA vs distance:",
      round(orders_df["rta"].corr(orders_df["customer_distance_miles"]), 4))
print(orders_df.groupby("hour")[["ptd", "rta"]].agg(["mean", "std"]).round(2))
print(orders_df.groupby("day_of_week")[["ptd", "rta"]].agg(["mean", "std"]).round(2))


# quantile of OTS + RTA within each period, with OTS uniform on the
# scenario range and RTA taken from the orders placed in that period
def ots_rta_quantile(rta_values, ots_range, level, n_grid=51):
    ots_grid = np.linspace(ots_range[0], ots_range[1], n_grid)
    sums = rta_values[:, None] + ots_grid[None, :]
    return np.quantile(sums, level)


rng = np.random.default_rng(6335)
promise_results = []

for scenario, ots_range in ots_scenarios.items():
    for level_name, level in service_levels.items():

        period_quantile = (
            orders_df.groupby(["week", "day_of_week", "hour"])["rta"]
            .apply(lambda x: ots_rta_quantile(x.values, ots_range, level))
            .rename("ots_rta_quantile").reset_index()
        )

        df = orders_df[["order_id", "week", "day_of_week", "hour", "ptd", "rta"]].merge(
            period_quantile, on=["week", "day_of_week", "hour"], how="left"
        )

        # OTD promise quoted to the customer, in whole minutes
        df["otd_promise"] = np.ceil(df["ots_rta_quantile"] + df["ptd"])

        # OTS promise: time the QF has to stage the order and still meet OTD
        df["ots_promise"] = df["otd_promise"] - df["ptd"] - df["rta"]

        # check: draw an OTS inside the scenario range, is the order on time?
        df["ots_draw"] = rng.uniform(ots_range[0], ots_range[1], len(df))
        df["on_time"] = df["ots_draw"] <= df["ots_promise"]

        df["scenario"] = scenario
        df["service_level"] = level_name
        promise_results.append(df)

promises_df = pd.concat(promise_results, ignore_index=True)


# summary of the OTS promise distribution for each scenario and service level
def summarise(group):
    lo, hi = ots_scenarios[group.name[0]]
    ots = group["ots_promise"]

    return pd.Series({
        "otd_mean": group["otd_promise"].mean(),
        "otd_p50": group["otd_promise"].quantile(0.50),
        "otd_p95": group["otd_promise"].quantile(0.95),
        "ots_mean": ots.mean(),
        "ots_std": ots.std(),
        "ots_min": ots.min(),
        "ots_p05": ots.quantile(0.05),
        "ots_p50": ots.quantile(0.50),
        "ots_p95": ots.quantile(0.95),
        "ots_max": ots.max(),
        "share_below_range": (ots < lo).mean(),
        "share_in_range": ots.between(lo, hi).mean(),
        "share_above_range": (ots > hi).mean(),
        "on_time_share": group["on_time"].mean(),
    })


ots_summary = (
    promises_df.groupby(["scenario", "service_level"], sort=False)
    .apply(summarise)
)

print(ots_summary.round(3).to_string())

# empirical OTS promise quantiles (distribution table) per scenario
quantile_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

ots_quantiles = (
    promises_df.groupby(["scenario", "service_level"], sort=False)["ots_promise"]
    .quantile(quantile_levels).unstack()
)

print(ots_quantiles.round(2).to_string())

# period dependence of the OTS promise (minimum service level)
base_df = promises_df[promises_df["service_level"] == "minimum"]

ots_by_weekday = (
    base_df.groupby(["scenario", "day_of_week"], sort=False)["ots_promise"]
    .mean().unstack()
)
ots_by_weekday.columns = day_names

ots_by_hour = (
    base_df.groupby(["scenario", "hour"], sort=False)["ots_promise"]
    .mean().unstack()
)

print(ots_by_weekday.round(2))
print(ots_by_hour.round(2))

ots_summary.round(4).to_csv("Output/T1f_ots_promise_summary.csv")
ots_quantiles.round(4).to_csv("Output/T1f_ots_promise_quantiles.csv")


# -------- Plots --------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

bins = np.arange(0, 20.5, 0.25)

for scenario, (lo, hi) in ots_scenarios.items():
    values = base_df.loc[base_df["scenario"] == scenario, "ots_promise"]
    axes[0].hist(values, bins=bins, alpha=0.5, density=True, label=scenario)

axes[0].set_xlabel("OTS promise (minutes)")
axes[0].set_ylabel("Density")
axes[0].set_title("OTS promise distribution (minimum service level)")
axes[0].legend()

for scenario in ots_scenarios:
    values = base_df.loc[base_df["scenario"] == scenario, "otd_promise"]
    counts = values.value_counts(normalize=True).sort_index()
    axes[1].plot(counts.index, counts.values * 100, "o-", label=scenario)

axes[1].set_xlabel("OTD promise (minutes)")
axes[1].set_ylabel("% of orders")
axes[1].set_title("OTD promise quoted to customers")
axes[1].legend()

plt.tight_layout()
plt.show()

# cumulative distribution of the OTS promise, all scenarios together
fig, ax = plt.subplots(figsize=(8, 4.8))

for scenario in ots_scenarios:
    values = np.sort(base_df.loc[base_df["scenario"] == scenario, "ots_promise"].values)
    ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=scenario)

ax.set_xlabel("OTS promise (minutes)")
ax.set_ylabel("Share of orders with OTS promise <= x")
ax.set_title("OTS promise CDF (minimum service level)")
ax.legend()

plt.tight_layout()
plt.savefig("Output/T1f_fig_ots_promise_cdf.png", dpi=150)
plt.show()