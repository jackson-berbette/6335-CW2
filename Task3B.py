import os
import pandas as pd
import numpy as np

# read relevant input files
sku_df = pd.read_csv("Data/QF_Facility_SKU_Master.csv")
supplier_df = pd.read_csv("Data/QF_Facility_Supplier_Sequence.csv")
modules_df = pd.read_csv("Data/QF_Facility_Storage_Modules.csv")
orders_df = pd.read_csv("Data/QF_Facility_Orders.csv")
order_lines_df = pd.read_csv("Data/QF_Facility_Order_Lines.csv")
targets_df = pd.read_csv("Output/task2a_inventory_autonomy_results.csv")

os.makedirs("Output", exist_ok=True)

scenarios = ["Fast", "Medium", "Slow"]
robustness_levels = [50, 90, 99]
zones = ["Ambient", "Chilled", "Frozen"]

# case-pack headroom rule chosen on the platform : "roundup", "overshoot" or "none"
headroom_rule = "roundup"

# extra slots kept on top of the calculated peak for shared and class storage,
# since real demand and receipts vary around the averages used here

safety_margin = 0.05

window_days = 112 # 2027-09-07 to 2027-12-27
window_start = pd.Timestamp("2027-09-07")

#------- inputs per product ---------
# volume per unit as stored: the case volume shared over its units
sku_df["unit_storage_l"] = sku_df["case_volume_l"] / sku_df["case_pack_units"]

storage_modules = modules_df[modules_df["module_category"] == "Storage"].set_index("zone")
sku_df["slot_volume_l"] = sku_df["temperature_zone"].map(storage_modules["usable_position_volume_l"])

# average units per day from the historical year
units_per_day = order_lines_df.groupby("sku_id")["quantity"].sum() / 365
sku_df["units_per_day"] = sku_df["sku_id"].map(units_per_day).fillna(0)

# velocity classes withing each zone: A = 80% of order lines, B = next 15%, C = last 5%
lines_per_sku = order_lines_df.groupby("sku_id").size()
sku_df["lines"] = sku_df["sku_id"].map(lines_per_sku).fillna(0)

sku_df = sku_df.sort_values(["temperature_zone", "lines"], ascending=[True, False])
sku_df["cum_share"] = (
    sku_df.groupby("temperature_zone")["lines"].cumsum()
    / sku_df.groupby("temperature_zone")["lines"].transform("sum")
)
sku_df["velocity_class"] = np.where(sku_df["cum_share"] <= 0.80, "A", np.where(sku_df["cum_share"] <= 0.95, "B", "C") )

print(sku_df.groupby(["temperature_zone", "velocity_class"]).size().unstack())

# -------- delivery calendars ----------
# delivery weekdays and lead time for the DC that serves each product
# Monday = 0 ... Sunday = 6; "every" = cycle length in weeks for weekly and longer cycles
calenders = {
    ("Fast", "Local"): {"weekdays": [0, 1, 2, 3, 4, 5, 6], "every": 1, "lead": 1},
    ("Fast", "Regional"): {"weekdays": [6, 3], "every": 1, "lead": 2},
    ("Fast", "National"): {"weekdays": [0], "every": 1, "lead": 4},
    ("Medium", "Local"): {"weekdays": [6, 3], "every": 1, "lead": 2},
    ("Medium", "Regional"): {"weekdays": [0], "every": 1, "lead": 5},
    ("Medium", "National"): {"weekdays": [0], "every": 2, "lead": 7},
    ("Slow", "Regional"): {"weekdays": [0], "every": 2, "lead": 7},
    ("Slow", "National"): {"weekdays": [0], "every": 4, "lead": 7},
}

anchor = pd.Timestamp("2026-09-07") # cycle anchor for 2- and 4-week Monday cycles
window_dates = pd.date_range(window_start, periods=window_days, freq="D")

def delivery_days(scenario, dc):
    cal = calenders[(scenario, dc)]
    is_delivery = []
    for date in window_dates:
        weeks_from_anchor = (date - anchor).days // 7
        is_delivery.append(date.dayofweek in cal["weekdays"] and weeks_from_anchor % cal["every"] == 0)
    return np.array(is_delivery), cal["lead"]

def stock_to_house(target, case_pack):
    if headroom_rule == "roundup":
        return np.ceil(target / case_pack) * case_pack
    if headroom_rule == "overshoot":
        return target + case_pack - 1
    return target

# -------- slot estimates per alternative --------
results = []
daily_rows = []

for scenario in scenarios:

    # DC that delivers each product in this scenario

    served_by = supplier_df.set_index("sku_id")["preferred_dc"]
    if scenario == "Slow":
        served_by = served_by.replace("Local", "Regional")

    for robustness in robustness_levels:

        df = sku_df.merge(
            targets_df[targets_df["scenario"] == scenario][["sku_id", "stock_" + str(robustness)]],
            on="sku_id"
        )
        df["target"] = df["stock_" + str(robustness)]
        df["house_units"] = stock_to_house(df["target"], df["case_pack_units"])
        df["served_by"] = df["sku_id"].map(served_by)

        # dedicated: each product keeps enough slots for its largest stock, at least one
        df["dedicated_slots"] = np.maximum(
            1, np.ceil(df["house_units"] * df["unit_storage_l"] / df["slot_volume_l"])
        )

        # shared: slots in use each day, following each product's stock through the window
        # stock starts at target; on a delivery day it is raised to the housed level less the 
        # demand during the lead time, and it falls by the average daily demand otherwise
        occupied = np.zeros((len(df), window_days))

        for dc in df["served_by"].unique():
            rows = np.where(df["served_by"] == dc)[0]
            is_delivery, lead = delivery_days(scenario, dc)

            on_hand = df["target"].values[rows].astype(float)
            peak_level = np.maximum(df["house_units"].values[rows] - df["units_per_day"].values[rows] * lead, 0)

            for day in range(window_days):
                if is_delivery[day]:
                    on_hand = np.maximum(on_hand, peak_level)
                occupied[rows, day] = np.ceil(
                    on_hand * df["unit_storage_l"].values[rows] / df["slot_volume_l"].values[rows]
                )
                on_hand = np.maximum(on_hand - df["units_per_day"].values[rows], 0)
            
        for zone in zones:
            in_zone = (df["temperature_zone"] == zone).values
            slots_per_module = storage_modules.loc[zone, "total_positions"]

            dedicated = int(df.loc[in_zone, "dedicated_slots"].sum())

            daily_occupied = occupied[in_zone].sum(axis=0)
            shared_peak = daily_occupied.max()

            # slots in use each day over the window, kept for the planning-horizon figure
            for day in range(window_days):
                daily_rows.append({"scenario" : scenario, "robustness" : robustness, "zone": zone,
                                    "date": window_dates[day].strftime("%Y-%m-%d"),
                                    "slots_in_use": int(daily_occupied[day])})
            shared = min(int(np.ceil(shared_peak * (1 + safety_margin))), dedicated)

            # class: shared inside each velocity class, each class sized to its own peak

            class_slots = 0
            class_modules = 0
            for velocity_class in ["A", "B", "C"]:
                in_class = in_zone & (df["velocity_class"] == velocity_class).values
                if in_class.sum() == 0:
                    continue
                peak = min(int(np.ceil(occupied[in_class].sum(axis=0).max() * (1 + safety_margin))),
                           int(df.loc[in_class, "dedicated_slots"].sum()))
                class_slots += peak
                class_modules += int(np.ceil(peak / slots_per_module))

            results .append({
                "scenario": scenario,
                "robustness": robustness,
                "zone": zone,
                "products": int(in_zone.sum()),
                "dedicated_slots": dedicated,
                "dedicated_modules": int(np.ceil(dedicated / slots_per_module)),
                "class_slots": class_slots,
                "class_modules": class_modules,
                "shared_slots": shared,
                "shared_modules": int(np.ceil(shared / slots_per_module)),
                "shared_peak_no_margin": int(shared_peak),
                "avg_slots_in_use": round(daily_occupied.mean(), 1),
                "start_volume_m3": round((df.loc[in_zone, "target"] * df.loc[in_zone, "unit_storage_l"]).sum() / 1000, 1)
            })

slots_df = pd.DataFrame(results)

print("Headroom rule:", headroom_rule, "| safety margin on shared/class:", safety_margin)
print(slots_df.to_string(index=False))

# check: start volume against the platform's "Space at Start" card
print(slots_df.groupby(["scenario", "robustness"])["start_volume_m3"].sum())

# module cost and energy per policy family
for family in ["dedicated", "class", "shared"]:
    slots_df[family + "_capex_usd"] = slots_df[family + "_modules"] * slots_df["zone"].map(storage_modules["capex_usd"])
    slots_df[family + "_energy_kwh_day"] = slots_df[family + "_modules"] * slots_df["zone"].map(storage_modules["energy_kwh_day"])

slots_df.to_csv("Output/T3_slot_estimates_" + headroom_rule + ".csv", index=False)
pd.DataFrame(daily_rows).to_csv("Output/T3_daily_slots_in_use_" + headroom_rule + ".csv", index=False)