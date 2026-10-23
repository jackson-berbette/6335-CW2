import pandas as pd
import numpy as np
import contextlib
import io
import math
import time

from pathlib import Path
from IPython.display import display

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import Task2A as t2a
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import Task1A as t1a

# read sku master data
sku_master = pd.read_csv("Data/QF_Facility_SKU_Master.csv")
sku_ordering = sku_master[
    [
        "sku_id",
        "case_pack_units",
        "opening_on_hand_units",
        "minimum_inbound_order_casepacks"
    ]
].copy()

# calculate qty of min inbound units
sku_ordering["minimum_inbound_units"] = (
    sku_ordering["case_pack_units"] * sku_ordering["minimum_inbound_order_casepacks"]
)

# Reproducible random results
rng = np.random.default_rng(42)

# Choose one SKU, scenario, and robustness level
test_sku = "P0001"
test_scenario = "Fast"
robustness = 90

# Get the target stock calculated in Part A
target_row = t2a.supply_aware_targets_df.loc[
    (t2a.supply_aware_targets_df["sku_id"] == test_sku)
    & (t2a.supply_aware_targets_df["scenario"] == test_scenario)
].iloc[0]

target_stock = int(target_row[f"stock_{robustness}"])

# Get SKU-specific ordering parameters
sku_row = sku_ordering.loc[
    sku_ordering["sku_id"] == test_sku
].iloc[0]

case_pack = int(sku_row["case_pack_units"])
opening_inventory = int(sku_row["opening_on_hand_units"])
minimum_order = int(sku_row["minimum_inbound_units"])

# Get the replenishment schedule for the chosen scenario
scenario_settings = t2a.scenario_config[(test_scenario, "Local")]
order_interval = scenario_settings["interval"]
lead_time = scenario_settings["lead"]

# function to simulate inventory given sku, replenishment scenario, robustness
def simulate_inventory(
    sku_id,
    scenario,
    robustness,
    days=365,
    seed=42
):
    rng = np.random.default_rng(seed)

    # retrieve target stock
    target_row = t2a.supply_aware_targets_df.loc[
        (t2a.supply_aware_targets_df["sku_id"] == sku_id)
        & (t2a.supply_aware_targets_df["scenario"] == scenario)
    ].iloc[0]

    target_stock = int(target_row[f"stock_{robustness}"])

    # retrieve SKU ordering parameters
    sku_row = sku_ordering.loc[
        sku_ordering["sku_id"] == sku_id
    ].iloc[0]

    case_pack = int(sku_row["case_pack_units"])
    min_order = int(sku_row["minimum_inbound_units"])
    on_hand = int(sku_row["opening_on_hand_units"])

    # retrieve the fitted demand model
    model = t1a.final_models[sku_id]

    if model is None:
        raise ValueError(f"No usable demand model for {sku_id}")

    # estimate expected daily demand
    forecast_data = pd.DataFrame({
        "time": (np.arange(1, days + 1) - 183) / 365,
        "day_of_week": (np.arange(1, days + 1) - 1) % 7
    })

    mu = np.asarray(model.predict(forecast_data), dtype=float)
    mu = np.maximum(mu, 0)

    # extract the Negative Binomial dispersion parameter
    params = model.params

    if hasattr(params, "index") and "alpha" in params.index:
        alpha = float(params["alpha"])
    else:
        alpha = getattr(
            getattr(model.model, "family", None),
            "alpha",
            None
        )

    if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            "Could not retrieve a positive Negative Binomial alpha. "
            "Inspect model.params and the model type before continuing."
        )

    # NB parameterization:
    # mean = mu, variance = mu + alpha * mu**2
    size = 1 / alpha
    probability = size / (size + mu)

    demand = rng.negative_binomial(size, probability)

    # retrieve the supplier sequence
    suppliers = t2a.get_scenario_sequence(sku_id, scenario)


 
    # --- Daily simulation state ---
    # Each pending event represents quantity that has not arrived:
    #   arrival: shipment expected at the facility
    #   escalate: remainder awaiting the next supplier attempt
    pending_events = []
    daily_records = []

    total_ordered = 0
    number_of_orders = 0
    escalations = 0
    total_demand = 0
    total_unmet = 0

    # Scenario-specific replenishment interval
    first_supplier = suppliers[0]
    scenario_key = (scenario, first_supplier)

    if scenario_key not in t2a.scenario_config:
        raise ValueError(
            f"Missing scenario configuration for {scenario_key}"
        )

    order_interval = t2a.scenario_config[scenario_key]["interval"]

    def schedule_supplier_attempt(quantity, supplier_index, attempt_day):
        """Schedule a supplier shipment and any escalation remainder."""
        nonlocal escalations

        if quantity <= 0:
            return

        # If no supplier remains, the quantity cannot be fulfilled
        if supplier_index >= len(suppliers):
            return

        dc = suppliers[supplier_index]
        settings = t2a.scenario_config.get((scenario, dc))

        if settings is None:
            raise ValueError(
                f"Missing scenario configuration for {(scenario, dc)}"
            )

        lead = int(settings["lead"])
        arrival_day = attempt_day + lead

        # Supplier either fills the complete request or 90% of it
        if rng.random() < 0.95:
            supplied = quantity
        else:
            supplied = math.floor(0.90 * quantity)
            escalations += 1

        remainder = quantity - supplied

        # Track the shipment as outstanding until it arrives
        if supplied > 0:
            pending_events.append({
                "day": arrival_day,
                "kind": "arrival",
                "quantity": supplied
            })

        # The remainder remains outstanding while being escalated.
        # It is attempted by the next supplier when this supplier's
        # lead-time window ends.
        if remainder > 0:
            pending_events.append({
                "day": arrival_day,
                "kind": "escalate",
                "quantity": remainder,
                "next_supplier_index": supplier_index + 1
            })

    for day in range(1, days + 1):

        # 1. Process all events due today.
        # Arrivals are processed before escalations on the same day.
        todays_events = [
            event for event in pending_events
            if event["day"] == day
        ]

        pending_events = [
            event for event in pending_events
            if event["day"] != day
        ]

        # Receive shipments first
        for event in todays_events:
            if event["kind"] == "arrival":
                on_hand += event["quantity"]

        # Then send unresolved remainders to the next supplier
        for event in todays_events:
            if event["kind"] == "escalate":
                schedule_supplier_attempt(
                    quantity=event["quantity"],
                    supplier_index=event["next_supplier_index"],
                    attempt_day=day
                )

        # 2. Realize demand and fulfill it from on-hand inventory
        todays_demand = int(demand[day - 1])
        fulfilled = min(on_hand, todays_demand)
        unmet = todays_demand - fulfilled
        on_hand -= fulfilled

        total_demand += todays_demand
        total_unmet += unmet

        # 3. Place an order at the scheduled opportunity
        order_qty = 0

        if (day - 1) % order_interval == 0:

            # Count all undelivered units, including quantities
            # awaiting escalation to another supplier.
            outstanding = sum(
                event["quantity"] for event in pending_events
            )

            inventory_position = on_hand + outstanding
            raw_qty = max(0, target_stock - inventory_position)

            if raw_qty > 0:
                # Round up to whole case packs
                order_qty = (
                    math.ceil(raw_qty / case_pack) * case_pack
                )

                # Enforce minimum inbound quantity
                order_qty = max(order_qty, min_order)
                order_qty = (
                    math.ceil(order_qty / case_pack) * case_pack
                )

                total_ordered += order_qty
                number_of_orders += 1

                # Begin the supplier sequence. Any undelivered
                # remainder will remain outstanding during escalation.
                schedule_supplier_attempt(
                    quantity=order_qty,
                    supplier_index=0,
                    attempt_day=day
                )

        # 4. Record end-of-day inventory and outstanding quantities
        outstanding_end = sum(
            event["quantity"] for event in pending_events
        )

        daily_records.append({
            "day": day,
            "demand": todays_demand,
            "fulfilled": fulfilled,
            "unmet_demand": unmet,
            "ending_on_hand": on_hand,
            "outstanding_orders": outstanding_end,
            "inventory_position": on_hand + outstanding_end,
            "order_quantity": order_qty,
            "stockout": unmet > 0
        })

    # End-of-horizon accounting
    ending_on_hand = on_hand

    ending_outstanding_units = sum(
        event["quantity"]
        for event in pending_events
        if event["kind"] == "arrival"
    )

    ending_inventory_position = (
        ending_on_hand + ending_outstanding_units
    )

    daily_df = pd.DataFrame(daily_records)


    # --- Summarize performance ---
    summary = {
        "sku_id": sku_id,
        "scenario": scenario,
        "robustness": robustness,
        "target_stock": target_stock,
        "opening_inventory": int(sku_row["opening_on_hand_units"]),
        "total_demand": total_demand,
        "total_unmet_demand": total_unmet,
        "fill_rate": (
            1 - total_unmet / total_demand
            if total_demand > 0 else 1.0
        ),
        "stockout_days": int(daily_df["stockout"].sum()),
        "average_on_hand": daily_df["ending_on_hand"].mean(),
        "average_inventory_position": daily_df[
            "inventory_position"
        ].mean(),
        "total_units_ordered": total_ordered,
        "number_of_orders": number_of_orders,
        "escalation_events": escalations,
        "ending_on_hand": ending_on_hand,
        "ending_outstanding_units": ending_outstanding_units,
        "ending_inventory_position": ending_inventory_position
    }

    return daily_df, pd.DataFrame([summary])


# ensure valid skus
target_skus = sorted(t2a.supply_aware_targets_df["sku_id"].unique())

missing_models = [
    sku for sku in target_skus
    if sku not in t1a.final_models or t1a.final_models[sku] is None
]

print(f"SKUs in target table: {len(target_skus)}")
print(f"SKUs with usable demand models: {len(target_skus) - len(missing_models)}")
print(f"SKUs missing demand models: {len(missing_models)}")

if missing_models:
    print("First missing SKUs:", missing_models[:20])

valid_skus = [sku for sku in target_skus if sku not in missing_models]

# run full simulation
scenarios = ["Fast", "Medium", "Slow"]
robustness_levels = [50, 90, 99]
seeds = range(5)

all_results = []
all_errors = []

total_runs = len(valid_skus) * len(scenarios) * len(robustness_levels) * len(seeds)

print(f"Starting {total_runs:,} simulation runs...")

completed = 0

for sku_id in valid_skus:
    for scenario in scenarios:
        for robustness in robustness_levels:
            for seed in seeds:
                try:
                    _, summary = simulate_inventory(
                        sku_id=sku_id,
                        scenario=scenario,
                        robustness=robustness,
                        days=365,
                        seed=seed
                    )

                    if isinstance(summary, pd.DataFrame):
                        result = summary.iloc[0].to_dict()
                    else:
                        result = dict(summary)

                    result["sku_id"] = sku_id
                    result["scenario"] = scenario
                    result["robustness"] = robustness
                    result["seed"] = seed

                    all_results.append(result)

                except Exception as e:
                    all_errors.append({
                        "sku_id": sku_id,
                        "scenario": scenario,
                        "robustness": robustness,
                        "seed": seed,
                        "error": str(e)
                    })

                completed += 1

    # print progress after each SKU.
    if len(all_results) + len(all_errors) > 0:
        if len(all_results) % 1000 < 45:
            print(
                f"Progress: {completed:,}/{total_runs:,} runs "
                f"({completed / total_runs:.1%})"
            )


simulation_results_df = pd.DataFrame(all_results)
simulation_errors_df = pd.DataFrame(all_errors)

print("\nSimulation complete.")

# aggregate data for the 5 seeds (simulations) for each sku-scenario-robustness combo
metrics_to_average = [
    "fill_rate",
    "stockout_days",
    "total_unmet_demand",
    "average_on_hand",
    "average_inventory_position",
    "total_units_ordered",
    "number_of_orders",
    "escalation_events"
]

metrics_to_average = [
    col for col in metrics_to_average
    if col in simulation_results_df.columns
]

group_cols = ["sku_id", "scenario", "robustness"]

simulation_summary_df = (
    simulation_results_df
    .groupby(group_cols)[metrics_to_average]
    .agg(["mean", "std"])
    .reset_index()
)


# save results data
simulation_results_df.to_csv(
    "Output/task2b_inventory_simulation_results.csv",
    index=False
)

simulation_summary_df.to_csv(
    "Output/task2b_inventory_simulation_summary.csv",
    index=False
)

if not simulation_errors_df.empty:
    simulation_errors_df.to_csv(
        "Output/task2b_inventory_simulation_errors.csv",
        index=False
    )