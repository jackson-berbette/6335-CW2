import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read relevant input files
sku_df = pd.read_csv("Data/QF_Facility_SKU_Master.csv")
supplier_df = pd.read_csv("Data/QF_Facility_Supplier_Sequence.csv")
modules_df = pd.read_csv("Data/QF_Facility_Storage_Modules.csv")
design_params_df = pd.read_csv("Data/QF_Facility_Design_Parameters.csv")

# outputs of Task 2a and Task 2b
targets_df = pd.read_csv("Output/task2a_inventory_autonomy_results.csv")

sim_df = pd.read_csv("Output/task2b_inventory_simulation_summary.csv", header=[0,1])
sim_df.columns = [
    col if stat.startswith("Unnamed") else col + "_" + stat
    for col, stat in sim_df.columns
]

os.makedirs("Output", exist_ok=True)

scenarios = ["Fast", "Medium", "Slow"]
robustness_levels = [50, 90, 99]
zones = ["Ambient", "Chilled", "Frozen"]

def get_parameter(name):
    return float(
        design_params_df.loc[design_params_df["parameter"] == name, "value"].iloc[0]
    )

full_fill_prob = get_parameter("full_fill_probability")
partial_fill = get_parameter("partial_fill_fraction")

# -------- Part c --------
# put target from 2a in the same long format as the 2b simulation
target_rows = []

for robustness in robustness_levels:
    df = targets_df[["sku_id", "scenario"]].copy()
    df["robustness"] = robustness
    df["autonomy_days"] = targets_df["autonomy_days_" + str(robustness)]
    df["target_stock"] = targets_df["stock_" + str(robustness)]
    target_rows.append(df)

targets_long_df = pd.concat(target_rows, ignore_index=True)

compare_df = sim_df.merge(targets_long_df, on=["sku_id", "scenario", "robustness"], how="left")

compare_df = compare_df.merge(
    sku_df[["sku_id", "temperature_zone", "unit_volume_l"]],
    on="sku_id",
    how="left"
).merge(
    supplier_df,
    on="sku_id",
    how="left"
)

print("Rows:", len(compare_df), "Missing targets:", compare_df["target_stock"].isna().sum())

# number of DCs each product can be supplied from in each scenario

def count_dcs(row):
    sequence = [row["preferred_dc"], row["second_dc"], row["third_dc"]]
    sequence = [dc for dc in sequence if pd.notna(dc)]

    if row["scenario"] == "Slow":
        sequence = ["Regional" if dc == "Local" else dc for dc in sequence]

    return len(list(dict.fromkeys(sequence)))

compare_df["num_dcs"] = compare_df.apply(count_dcs, axis=1)

# exposure to an unfillable supplier order:
# an order is left short only if every DC in the sequence misses the full fill,
# and each miss passes on (1 - partial fill) of what it was given
compare_df["prob_order_unfillable"] = (1 - full_fill_prob) ** compare_df["num_dcs"]
compare_df["unfilled_fraction"] = ((1 - full_fill_prob) * (1 - partial_fill)) ** compare_df["num_dcs"]

compare_df["unfillable_orders"] = compare_df["number_of_orders_mean"] * compare_df["prob_order_unfillable"]
compare_df["unfilled_units"] = compare_df["total_units_ordered_mean"] * compare_df["unfilled_fraction"]

# space occupied by stock: liters held on average and at the target level
compare_df["avg_stock_litres"] = compare_df["average_on_hand_mean"] * compare_df["unit_volume_l"]
compare_df["target_stock_litres"] = compare_df["target_stock"] * compare_df["unit_volume_l"]

# summary for each replenishment scenario and robustness level
summary_df = (
    compare_df.groupby(["scenario", "robustness"])
    .agg(mean_autonomy_days=("autonomy_days", "mean"),
         target_stock_units=("target_stock", "sum"),
         units_ordered_per_year=("total_units_ordered_mean", "sum"),
         orders_per_year=("number_of_orders_mean", "sum"),
         avg_inventory_units=("average_on_hand_mean", "sum"),
         avg_stock_litres=("avg_stock_litres", "sum"),
         target_stock_litres=("target_stock_litres", "sum"),
         partial_fills_per_year=("escalation_events_mean", "sum"),
         unfillable_orders_per_year=("unfillable_orders", "sum"),
         unfilled_units_per_year=("unfilled_units", "sum"),
         mean_fill_rate=("fill_rate_mean", "mean"),
         mean_stockout_days=("stockout_days_mean", "mean"),
         lost_units_per_year=("total_unmet_demand_mean", "sum"))
         .reindex(pd.MultiIndex.from_product([scenarios, robustness_levels], names=["scenario", "robustness"]))
)

summary_df["avg_order_units"] = summary_df["units_ordered_per_year"] / summary_df["orders_per_year"]
summary_df["inbound_orders_per_week"] = summary_df["orders_per_year"] / 52
summary_df["inventory_days_of_cover"] = (
    summary_df["avg_inventory_units"] / (summary_df["units_ordered_per_year"] / 365)
)
summary_df["unfillable_share_of_orders"] = (
    summary_df["unfillable_orders_per_year"] / summary_df["orders_per_year"]
)

print(summary_df.round(3).T.to_string())

# autonomy target and order frequency by preferred DC
by_dc_df = (
    compare_df.groupby(["scenario", "robustness", "preferred_dc"])
    .agg(products=("sku_id", "size"),
         autonomy_days=("autonomy_days", "mean"),
         orders_per_product=("number_of_orders_mean", "mean"),
         order_units=("total_units_ordered_mean", "sum"),
         orders=("number_of_orders_mean", "sum"),
         avg_inventory_units=("average_on_hand_mean", "sum"),
         fill_rate=("fill_rate_mean", "mean"),
         prob_order_unfillable=("prob_order_unfillable", "mean"))
)

by_dc_df["avg_order_units"] = by_dc_df["order_units"] / by_dc_df["orders"]
by_dc_df = by_dc_df.drop(columns=["order_units", "orders"])

print(by_dc_df.round(4).to_string())

# space by temperature zone: litres -> storage positions -> modules -> floor area
storage_modules = modules_df[modules_df["module_category"] == "Storage"].set_index("zone")
storage_modules["module_area_sqft"] = storage_modules["width_ft"] * storage_modules["depth_ft"]

space_rows = []

for (scenario, robustness), group in compare_df.groupby(["scenario", "robustness"]):
    for zone in zones:
        zone_group = group[group["temperature_zone"] == zone]
        module = storage_modules.loc[zone]

        avg_litres = zone_group["avg_stock_litres"].sum()
        target_litres = zone_group["target_stock_litres"].sum()

        # positions needed if stock were pooled, and with one position minimum per product

        avg_positions = avg_litres / module["usable_position_volume_l"]
        target_positions = target_litres / module["usable_position_volume_l"]
        slotted_positions = np.ceil(
            zone_group["target_stock_litres"] / module["usable_position_volume_l"]
        ).clip(lower=1).sum()

        modules_needed = np.ceil(slotted_positions / module["total_positions"])

        space_rows.append({
            "scenario": scenario,
            "robustness": robustness,
            "zone": zone,
            "products": len(zone_group),
            "avg_stock_litres": avg_litres,
            "target_stock_litres": target_litres,
            "avg_stock_positions": avg_positions,
            "target_stock_positions": target_positions,
            "slotted_positions": slotted_positions,
            "storage_modules": modules_needed,
            "module_floor_sqft": modules_needed * module["module_area_sqft"]
        })

space_df = pd.DataFrame(space_rows)

space_total_df = (
    space_df.groupby(["scenario", "robustness"])
    [["avg_stock_positions", "target_stock_positions", "slotted_positions",
      "storage_modules", "module_floor_sqft"]]
    .sum()
    .reindex(summary_df.index)
)

summary_df = summary_df.join(space_total_df)

print(space_df.round(1).to_string(index=False))
print(space_total_df.round(1).to_string())

# exposure to unfillable supplier orders by number of DCs in the sequence
exposure_df = (
    compare_df[compare_df["robustness"] == 90]
    .groupby(["scenario", "num_dcs"])
    .agg(products=("sku_id", "size"),
         prob_order_unfillable=("prob_order_unfillable", "mean"),
         orders_per_year=("number_of_orders_mean", "sum"),
         partial_fills_per_year=("escalation_events_mean", "sum"),
         unfillable_orders_per_year=("unfillable_orders", "sum"),
         unfilled_units_per_year=("unfilled_units", "sum"))
)

# check: simulated partial fills against the analytical 1 - full fill probability
exposure_df["simulated_partial_fill_share"] = (
    exposure_df["partial_fills_per_year"] / exposure_df["orders_per_year"]
)

print(exposure_df.round(4).to_string())

summary_df.round(4).to_csv("Output/T2c_alternative_comparison.csv")
by_dc_df.round(4).to_csv("Output/T2c_comparison_by_dc.csv")
space_df.round(2).to_csv("Output/T2c_space_by_zone.csv", index=False)
exposure_df.round(5).to_csv("Output/T2c_supplier_exposure.csv")

# -------- Plots --------
colors = {"Fast": "tab:blue", "Medium": "tab:orange", "Slow": "tab:green"}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for scenario in scenarios:
    data = summary_df.loc[scenario]

    axes[0].plot(data["avg_inventory_units"] / 1000, data["mean_fill_rate"] * 100,
                 "o-", color=colors[scenario], label=scenario)
    for robustness in robustness_levels:
        axes[0].annotate(f"{robustness}%",
                         (data.loc[robustness, "avg_inventory_units"] / 1000,
                          data.loc[robustness, "mean_fill_rate"] * 100),
                         textcoords="offset points", xytext=(5, -10), fontsize=8)

axes[0].set_xlabel("Average inventory (thousand units)")
axes[0].set_ylabel("Mean fill rate (%)")
axes[0].set_title("Inventory vs service")
axes[0].legend()

x = np.arange(len(robustness_levels))
width = 0.25

for i, scenario in enumerate(scenarios):
    data = summary_df.loc[scenario]
    axes[1].bar(x + (i - 1) * width, data["slotted_positions"], width,
                color=colors[scenario], label=scenario)

axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{r}%" for r in robustness_levels])
axes[1].set_xlabel("Robustness level")
axes[1].set_ylabel("Storage positions at target stock")
axes[1].set_title("Space occupied by stock")
axes[1].set_ylim(0, summary_df["slotted_positions"].max() * 1.2)
axes[1].legend(ncol=3, loc="upper center")

for i, scenario in enumerate(scenarios):
    data = summary_df.loc[scenario]
    axes[2].bar(x + (i - 1) * width, data["inbound_orders_per_week"], width,
                color=colors[scenario], label=scenario)

axes[2].set_xticks(x)
axes[2].set_xticklabels([f"{r}%" for r in robustness_levels])
axes[2].set_xlabel("Robustness level")
axes[2].set_ylabel("Inbound orders per week")
axes[2].set_title("Inbound frequency")
axes[2].set_ylim(0, summary_df["inbound_orders_per_week"].max() * 1.2)
axes[2].legend(ncol=3, loc="upper center")

plt.tight_layout()
plt.savefig("Output/T2c_fig_comparison.png", dpi=150)
plt.show()