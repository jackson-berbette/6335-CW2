import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# read relevant input files
orders_df = pd.read_csv("Data/QF_Facility_Orders.csv")
order_lines_df = pd.read_csv("Data/QF_Facility_Order_Lines.csv")


# -------- Part e --------
# count lines and units in each temp zone for every order
zones = ["Ambient", "Chilled", "Frozen"]

zone_lines_df = (
    order_lines_df.pivot_table(index="order_id", columns="temperature_zone",
                               values="quantity", aggfunc="size", fill_value=0)
    .reindex(columns=zones, fill_value=0)
)

zone_units_df = (
    order_lines_df.pivot_table(index="order_id", columns="temperature_zone",
                               values="quantity", aggfunc="sum", fill_value=0)
    .reindex(columns=zones, fill_value=0)
)

# one row per order: which zones it touches, how many, and its combination
order_zones_df = pd.DataFrame({"order_id": zone_lines_df.index})

for zone in zones:
    order_zones_df["has_" + zone] = (zone_lines_df[zone].values > 0)
    order_zones_df[zone + "_lines"] = zone_lines_df[zone].values

order_zones_df["lines"] = zone_lines_df.sum(axis=1).values
order_zones_df["num_zones"] = order_zones_df[["has_Ambient", "has_Chilled", "has_Frozen"]].sum(axis=1)
order_zones_df["multi_zone"] = order_zones_df["num_zones"] > 1


combo_names = []

for i, row in order_zones_df.iterrows():
    present = [zone for zone in zones if row["has_" + zone]]
    combo_names.append(" + ". join(present))

order_zones_df["zone_combo"] = combo_names

# add weekday and hour of order
order_zones_df = order_zones_df.merge(
    orders_df[["order_id", "day_of_week", "placement_timestamp"]],
    on="order_id",
    how="left"
)

order_zones_df["hour"] = pd.to_datetime(order_zones_df["placement_timestamp"]).dt.hour


n_orders = len(order_zones_df)
print("Orders:", n_orders)

# probability of each zone combination
combo_df = (
    order_zones_df.groupby("zone_combo")
    .agg(orders=("order_id", "size"),
         mean_lines=("lines", "mean"))
         .reset_index()
)

combo_df["probability"] = combo_df["orders"] / n_orders
combo_df = combo_df.sort_values("probability", ascending=False).reset_index(drop=True)

print(combo_df)

# number of zones per order
num_zones_df = (
    order_zones_df.groupby("num_zones")
    .agg(orders=("order_id", "size"),
         mean_lines=("lines", "mean"))
         .reset_index()
)

num_zones_df["probability"] = num_zones_df["orders"] / n_orders

print(num_zones_df)
print("Share of orders spanning more than one zone:", order_zones_df["multi_zone"].mean())


# how often each zone appears, and its share of lines and units
zone_summary = []

for zone in zones:

    has_zone = order_zones_df["has_" + zone]

    row = {
        "zone": zone,
        "prob_order_has_zone": has_zone.mean(),
        "share_of_lines": zone_lines_df[zone].sum() / zone_lines_df.values.sum(),
        "share_of_units": zone_units_df[zone].sum() / zone_units_df.values.sum(),
        "mean_lines_when_present": order_zones_df.loc[has_zone, zone +"_lines"].mean()
    }

    zone_summary.append(row)

zone_summary_df = pd.DataFrame(zone_summary)

print(zone_summary_df)
print("Share of orders with any cold item (chilled or frozen):",
      (order_zones_df["has_Chilled"] | order_zones_df["has_Frozen"]).mean())


# are zones chosen independently of each other?
# expected probability of each combination if zones were independent
p_zone = {zone: order_zones_df["has_" + zone].mean() for zone in zones}

independence = []

for i, row in combo_df.iterrows():

    expected_prob = 1.0

    for zone in zones:
        if zone in row["zone_combo"].split(" + "):
            expected_prob = expected_prob * p_zone[zone]
        else:
            expected_prob = expected_prob * (1 - p_zone[zone])

    independence.append({
        "zone_combo": row["zone_combo"],
        "observed_prob": row["probability"],
        "expected_if_independent": expected_prob,
        "observed_to_expected": row["probability"] / expected_prob
    })

independence_df = pd.DataFrame(independence)

chi2_test = stats.chisquare(
    combo_df["orders"],
    independence_df["expected_if_independent"] * n_orders / independence_df["expected_if_independent"].sum()
)

print(independence_df)
print("Chi-square independence test p-value:", chi2_test.pvalue)

# zone span by basket size
by_basket_size_df = (
    order_zones_df[order_zones_df["lines"] <= 12]
    .groupby("lines")
    .agg(orders=("order_id", "size"),
         share_multi_zone=("multi_zone", "mean"),
         mean_zone=("num_zones", "mean"),
         share_with_chilled=("has_Chilled", "mean"),
         share_with_frozen=("has_Frozen", "mean"))
    .reset_index()
)

print(by_basket_size_df)

# zone span by weekday and by hour of the day
by_weekday_df = (
    order_zones_df.groupby("day_of_week")
    .agg(orders=("order_id", "size"),
         share_multi_zone=("multi_zone", "mean"),
         share_with_chilled=("has_Chilled", "mean"),
         share_with_frozen=("has_Frozen", "mean"))
    .reset_index()
)

by_hour_df = (
    order_zones_df.groupby("hour")
    .agg(orders=("order_id", "size"),
        share_multi_zone=("multi_zone", "mean"),
        share_with_chilled=("has_Chilled", "mean"),
        share_with_frozen=("has_Frozen", "mean"))
    .reset_index()
)

print(by_weekday_df)
print(by_hour_df)

# plot zone combinations and zone span by basket size
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].barh(combo_df["zone_combo"], combo_df["probability"] * 100)
axes[0].invert_yaxis()
axes[0].set_xlim(0, combo_df["probability"].max() * 100 * 1.15)
axes[0].set_xlabel("% of orders")
axes[0].set_title("Temperature zone combinations")

for i, value in enumerate(combo_df["probability"] * 100):
    axes[0].text(value + 0.5, i, f"{value:.1f}%", va="center")

axes[1].plot(by_basket_size_df["lines"], by_basket_size_df["share_multi_zone"] * 100,
             "o-", label="More than one zone")

axes[1].plot(by_basket_size_df["lines"], by_basket_size_df["share_with_chilled"] * 100,
             "s-", label="Has chilled")

axes[1].plot(by_basket_size_df["lines"], by_basket_size_df["share_with_frozen"] * 100,
             "^-", label="Has frozen")

axes[1].set_xlabel("Order lines per order")
axes[1].set_ylabel("% of orders")
axes[1].set_title("Zone span by basket size")
axes[1].legend()

plt.tight_layout()
plt.show()