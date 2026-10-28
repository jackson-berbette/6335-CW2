import pandas as pd
import numpy as np
import contextlib
import io
import os
from scipy.stats import nbinom
from IPython.display import display


with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import Task1A as t1

supplier_df = pd.read_csv("Data/QF_Facility_Supplier_Sequence.csv")

# maximum gap in days between shipments
# Assumptions: conservative 4-day gap for twice-weekly, 30-day for monthly
# Scenario/DC configuration. Intervals are calendar-day approximations.
scenario_config = {
    ("Fast", "Local"):    {"interval": 1,  "lead": 2},
    ("Fast", "Regional"): {"interval": 4,  "lead": 3},
    ("Fast", "National"): {"interval": 7,  "lead": 5},
    ("Medium", "Local"):    {"interval": 4,  "lead": 3},
    ("Medium", "Regional"): {"interval": 7,  "lead": 6},
    ("Medium", "National"): {"interval": 14, "lead": 8},
    ("Slow", "Regional"): {"interval": 14, "lead": 8},
    ("Slow", "National"): {"interval": 30, "lead": 8},
}


def get_scenario_sequence(sku_id, scenario):
    row = supplier_df.loc[supplier_df["sku_id"] == sku_id].iloc[0]

    sequence = [
        row["preferred_dc"],
        row["second_dc"],
        row["third_dc"]
    ]
    sequence = [dc for dc in sequence if pd.notna(dc)]

    if scenario == "Slow":
        sequence = [
            "Regional" if dc == "Local" else dc
            for dc in sequence
        ]

    # Preserve sequence order and remove duplicate DC types.
    return list(dict.fromkeys(sequence))


def next_delivery_day(earliest_day, interval):
    """Next eligible delivery day under a repeating interval approximation."""
    return earliest_day + ((-earliest_day) % interval)


def simulate_effective_horizon(
    sku_id,
    scenario,
    n_simulations=10000,
    random_state=42
):
    sequence = get_scenario_sequence(sku_id, scenario)

    if not sequence:
        raise ValueError(f"No supplier sequence for {sku_id}")

    primary = sequence[0]
    primary_config = scenario_config[(scenario, primary)]

    base_horizon = (
        primary_config["interval"] + primary_config["lead"]
    )

    rng = np.random.default_rng(random_state)
    effective_horizons = np.zeros(n_simulations, dtype=int)

    for sim in range(n_simulations):
        # Treat the order as 1 unit of requested quantity.
        # The quantity scale does not affect the arrival timing.
        remaining_fraction = 1.0

        # Initial delivery occurs after the primary DC's protection horizon.
        arrival_day = base_horizon

        for dc_index, dc in enumerate(sequence):
            dc_interval, dc_lead = (
                scenario_config[(scenario, dc)]["interval"],
                scenario_config[(scenario, dc)]["lead"]
            )

            # The primary DC's initial arrival is already scheduled.
            if dc_index > 0:
                earliest = arrival_day + dc_lead
                arrival_day = next_delivery_day(earliest, dc_interval)

            full_fill = rng.random() < 0.95

            if full_fill:
                remaining_fraction = 0.0
                break

            # DC supplies 90% of the remaining requested quantity.
            remaining_fraction *= 0.10

            # Escalate any remainder to the next DC.
            if remaining_fraction < 1e-8:
                remaining_fraction = 0.0
                break

        # If all DCs were tried, use the last arrival as the completion
        # time for this approximation, even if a tiny remainder persists.
        effective_horizons[sim] = max(base_horizon, arrival_day)

    return base_horizon, effective_horizons


def demand_targets_with_supply_uncertainty(
    sku_id,
    scenario,
    n_simulations=10000,
    random_state=42
):
    model = t1.final_models.get(sku_id)

    if sku_id not in t1.final_models:
        raise ValueError(f"No model entry for {sku_id}")

    if model is None:
        return {
            "sku_id": sku_id,
            "scenario": scenario,
            "base_horizon_days": np.nan,
            "autonomy_days_50": np.nan,
            "autonomy_days_90": np.nan,
            "autonomy_days_99": np.nan,
            "stock_50": np.nan,
            "stock_90": np.nan,
            "stock_99": np.nan
        }

    alpha = float(model.params["alpha"])
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError(f"Invalid dispersion for {sku_id}")

    base_horizon, effective_horizons = simulate_effective_horizon(
        sku_id,
        scenario,
        n_simulations=n_simulations,
        random_state=random_state
    )

    max_horizon = int(effective_horizons.max())

    # Forecast daily means into the future.
    weekday_by_day = (
        t1.orders_df[["day", "day_of_week"]]
        .drop_duplicates("day")
        .set_index("day")["day_of_week"]
        .sort_index()
    )

    last_day = int(weekday_by_day.index.max())
    weekdays = sorted(weekday_by_day.unique())
    last_weekday_index = weekdays.index(weekday_by_day.loc[last_day])

    future_days = np.arange(last_day + 1, last_day + max_horizon + 1)
    future_weekdays = [
        weekdays[(last_weekday_index + d - last_day) % len(weekdays)]
        for d in future_days
    ]

    future_df = pd.DataFrame({
        "day": future_days,
        "day_of_week": future_weekdays,
        "time": (future_days - 183) / 365
    })

    means = np.maximum(
        np.asarray(model.predict(future_df), dtype=float),
        0
    )

    size = 1 / alpha
    probabilities = size / (size + means)

    # Simulate cumulative demand for each path, then select demand
    # at that path's supplier-dependent horizon.
    rng = np.random.default_rng(random_state + 1)
    cumulative_demand = np.zeros(n_simulations, dtype=int)

    # Generate daily demand draws in batches by day.
    demand_paths = np.zeros(
        (n_simulations, max_horizon),
        dtype=np.int32
    )

    for day_index in range(max_horizon):
        if means[day_index] > 0:
            demand_paths[:, day_index] = rng.negative_binomial(
                size,
                probabilities[day_index],
                size=n_simulations
            )

    cumulative_paths = np.cumsum(demand_paths, axis=1)

    demand_exposure = cumulative_paths[
        np.arange(n_simulations),
        effective_horizons - 1
    ]

    return {
        "sku_id": sku_id,
        "scenario": scenario,
        "base_horizon_days": base_horizon,
        "autonomy_days_50": int(np.ceil(np.quantile(
            effective_horizons, 0.50
        ))),
        "autonomy_days_90": int(np.ceil(np.quantile(
            effective_horizons, 0.90
        ))),
        "autonomy_days_99": int(np.ceil(np.quantile(
            effective_horizons, 0.99
        ))),
        "stock_50": int(np.ceil(np.quantile(demand_exposure, 0.50))),
        "stock_90": int(np.ceil(np.quantile(demand_exposure, 0.90))),
        "stock_99": int(np.ceil(np.quantile(demand_exposure, 0.99)))
    }

# apply to all SKUs
all_supply_target_rows = []
supply_target_failures = []

scenarios = ["Fast", "Medium", "Slow"]
all_skus = sorted(supplier_df["sku_id"].unique())

for i, sku_id in enumerate(all_skus, start=1):
    for scenario in scenarios:
        try:
            result = demand_targets_with_supply_uncertainty(
                sku_id=sku_id,
                scenario=scenario,
                n_simulations=2000,
                random_state=42
            )
            all_supply_target_rows.append(result)

        except Exception as e:
            supply_target_failures.append({
                "sku_id": sku_id,
                "scenario": scenario,
                "error": str(e)
            })

supply_aware_targets_df = pd.DataFrame(all_supply_target_rows)
supply_target_failures_df = pd.DataFrame(supply_target_failures)

print("\nCompleted")
print("SKUs in supplier file:", len(all_skus))
print("Target rows:", len(supply_aware_targets_df))
print("Failed SKU-scenario combinations:", len(supply_target_failures_df))

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
display(supply_aware_targets_df.head(10))
display(supply_target_failures_df.head(10))


# store output
supply_aware_targets_df.to_csv(
    "Output/task2a_inventory_autonomy_results.csv", index=False
)

# generate 9 csvs for evaluator
scenarios = ["Fast", "Medium", "Slow"]
robustness_levels = [50, 90, 99]

policy_id = 1

for scenario in scenarios:

    scenario_df = supply_aware_targets_df[
        supply_aware_targets_df["scenario"] == scenario
    ]

    for robustness in robustness_levels:

        service_target = robustness / 100

        output_df = scenario_df[
            ["sku_id", f"stock_{robustness}"]
        ].copy()

        # Add required metadata
        output_df["policy_id"] = f"POLICY_{policy_id:02d}"
        output_df["scenario_id"] = scenario.upper()
        output_df["service_target"] = service_target

        # Effective day: first day the policy applies
        output_df["effective_day"] = "2027-09-07"

        # Rename inventory target column
        output_df = output_df.rename(columns={
            f"stock_{robustness}": "target_stock_units"
        })

        # Arrange columns in required order
        output_df = output_df[
            [
                "policy_id",
                "scenario_id",
                "service_target",
                "sku_id",
                "effective_day",
                "target_stock_units"
            ]
        ]

        output_df = output_df.sort_values("sku_id")

        # Validate before saving
        assert len(output_df) == 2000, (
            f"Expected 2000 SKUs for {scenario} {robustness}%"
        )
        assert output_df["sku_id"].nunique() == 2000
        assert not output_df["target_stock_units"].isna().any()

        # Export
        filename = (
            f"task2_{scenario.lower()}_{robustness}.csv"
        )

        filepath = os.path.join("Output", filename)

        output_df.to_csv(filepath, index=False)

        print(f"Saved {filepath}: {len(output_df)} rows")

        policy_id += 1