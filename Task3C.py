import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read relevant input files
slots_df = pd.read_csv("Output/T3_slot_estimates_roundup.csv")          
modules_df = pd.read_csv("Data/QF_Facility_Storage_Modules.csv")
travel_df = pd.read_csv("Data/T3_platform_travel.csv")

os.makedirs("Output", exist_ok=True)

scenarios = ["Fast", "Medium", "Slow"]
robustness_levels = [50, 90, 99]
families = ["dedicated", "class", "shared"]
family_policies = {"dedicated": "1-3", "class": "4-6", "shared": "7-9"}

storage_modules = modules_df[modules_df["module_category"] == "Storage"].set_index("zone")
slots_df["slot_volume_l"] = slots_df["zone"].map(storage_modules["usable_position_volume_l"])

# -------- Part c --------
# storage space and slot utilization for each alternative and policy family
space_rows = []

for (scenario, robustness), group in slots_df.groupby(["scenario", "robustness"], sort=False):
    for family in families:

        slots = group[family + "_slots"].sum()
        capacity_m3 = (group[family + "_slots"] * group["slot_volume_l"]).sum() / 1000

        space_rows.append({
            "scenario": scenario,
            "robustness": robustness,
            "policies": family_policies[family],
            "slots": slots,
            "modules": group[family + "_modules"].sum(),
            "capex_usd": group[family + "_capex_usd"].sum(),
            # slot occupancy: average slots holding stock over the 16 weeks / slots provided
            "slot_occupancy": group["avg_slots_in_use"].sum() / slots,
            # cube fill: stock volume at target / usable volume of the slots provided
            "cube_fill_at_target": group["start_volume_m3"].sum() / capacity_m3
        })

space_df = pd.DataFrame(space_rows)
print(space_df.round(3).to_string(index=False))

# travel per pick line for every policy at the 90% target (Travel ft/line on the platform's reference layout)
travel_90 = (
    travel_df[travel_df["robustness"] == 90]
    .pivot(index=["policy_no", "policy"], columns="scenario", values="travel_ft_per_line")
    .reindex(columns=scenarios)
)

# change against the mandatory Product · Dedicated · COL (policy 1) in the same scenario
travel_vs_p1 = travel_90 / travel_90.xs(1, level="policy_no").iloc[0] - 1

print(travel_90.round(1).to_string())
print(travel_vs_p1.round(3).to_string())

# effect of the service target: mandatory policy 1 in every alternative
p1_df = travel_df[travel_df["policy_no"] == 1][["scenario", "robustness", "travel_ft_per_line"]].merge(
    space_df[space_df["policies"] == "1-3"][["scenario", "robustness", "slots", "modules", "slot_occupancy", "cube_fill_at_target"]],
    on=["scenario", "robustness"]
)

print(p1_df.round(3).to_string(index=False))

space_df.round(4).to_csv("Output/T3c_space_and_utilization.csv", index=False)
travel_90.round(1).to_csv("Output/T3c_travel_by_policy.csv")
p1_df.round(4).to_csv("Output/T3c_service_target_effect.csv", index=False)

# -------- Plots --------
colors = {"Fast": "tab:blue", "Medium": "tab:orange", "Slow": "tab:green"}

# figure 1: travel per pick line by policy and scenario (90% target)
fig, ax = plt.subplots(figsize=(11, 4.2))

labels = [str(no) + " " + name for no, name in travel_90.index]
x = np.arange(len(labels))
width = 0.27

for i, scenario in enumerate(scenarios):
    values = travel_90[scenario].values
    ax.bar(x + (i - 1) * width, np.nan_to_num(values), width, color=colors[scenario], label=scenario)
    for xi, v in zip(x + (i - 1) * width, values):
        if np.isnan(v):
            ax.text(xi, 2, "n/c", ha="center", va="bottom", fontsize=7, rotation=90, color="grey")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Travel per pick line (ft)")
ax.set_title("Picker travel by storage policy and replenishment scenario (90% target, reference layout)", loc="left", fontsize=10)
ax.legend(ncol=3, frameon=False)
ax.grid(axis="y", color="#e5e7eb")

plt.tight_layout()
plt.savefig("Output/T3c_fig_travel_by_policy.png", dpi=170)
plt.show()

# figure 2: effect of service target and scenario on slots and travel (policy 1)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for scenario in scenarios:
    data = p1_df[p1_df["scenario"] == scenario].sort_values("robustness")
    axes[0].plot(data["robustness"], data["slots"], "o-", color=colors[scenario], label=scenario)
    axes[1].plot(data["robustness"], data["travel_ft_per_line"], "o-", color=colors[scenario], label=scenario)

for family, style in zip(families, ["-", "--", ":"]):
    data = space_df[(space_df["robustness"] == 90) & (space_df["policies"] == family_policies[family])]
    axes[2].plot(scenarios, data["slot_occupancy"] * 100, "o" + style, color="black",
                 label="Policies " + family_policies[family])


axes[0].set_title("Slots needed (policy 1)", loc="left", fontsize=10)
axes[0].set_ylabel("Storage slots (all zones)")
axes[1].set_title("Travel per pick line (policy 1)", loc="left", fontsize=10)
axes[1].set_ylabel("ft per line")
for ax in axes[:2]:
    ax.set_xlabel("Service target (%)")
    ax.set_xticks(robustness_levels)
    ax.legend(frameon=False)
    ax.grid(color="#e5e7eb")
axes[2].set_title("Slot occupancy by policy family (90% target)", loc="left", fontsize=10)
axes[2].set_ylabel("Average slots in use / slots provided (%)")
axes[2].legend(frameon=False)
axes[2].grid(color="#e5e7eb")

plt.tight_layout()
plt.savefig("Output/T3c_fig_target_and_scenario.png", dpi=170)
plt.show()


