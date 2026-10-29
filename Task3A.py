import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

# read relevant input files
sku_df = pd.read_csv("Data/QF_Facility_SKU_Master.csv")
order_lines_df = pd.read_csv("Data/QF_Facility_Order_Lines.csv")
slots_df = pd.read_csv("Output/T3_slot_estimates_roundup.csv")

os.makedirs("Output", exist_ok=True)

zones = ["Ambient", "Chilled", "Frozen"]

# -------- Part a: what drives the choice of storage policy --------
# pick concentration: share of order lines taken by the most-picked products
lines_per_sku = (
    order_lines_df.groupby("sku_id").size()
    .reindex(sku_df["sku_id"], fill_value=0)
    .sort_values(ascending=False)
)
cum_share = lines_per_sku.cumsum() / lines_per_sku.sum()

print("Share of order lines from the top 20 products:", round(cum_share.iloc[19], 3))
print("Share of order lines from the top 400 products (20%):", round(cum_share.iloc[399], 3))

# pick concentration withing each temperature zone - top 20% of the zone's products
zone_rows = []

for zone in zones:
    zone_skus = sku_df.loc[sku_df["temperature_zone"] == zone, 'sku_id']
    zone_lines = lines_per_sku[lines_per_sku.index.isin(zone_skus)].sort_values(ascending=False)
    top_20 = int(len(zone_lines) * 0.20)

    zone_rows.append({
        "zone": zone,
        "products": len(zone_lines),
        "share_of_lines_top_20pct": zone_lines.iloc[:top_20].sum() / zone_lines.sum(),
        "median_shelf_life_days": sku_df.loc[sku_df["temperature_zone"] == zone, "shelf_life_days"].median()
    })

drivers_df = pd.DataFrame(zone_rows)
print(drivers_df.round(3))

# slots per product - task 3b dedicated estimate, fast / 90%
fast_90 = slots_df[(slots_df["scenario"] == "Fast") & (slots_df["robustness"] == 90)]
print("Slots for 2,000 products at Fast / 90%:" , fast_90["dedicated_slots"].sum(),
      "-> slots per product:", round(fast_90["dedicated_slots"].sum() / fast_90["products"].sum(), 2))

drivers_df.round(4).to_csv("Output/T3a_policy_drivers.csv", index=False)

# ------ Plots -------
# figure 1: the 18 facet combinations and what happened to each
# S = selected, R = rejected (supported but duplicated another), U = not supported by the platform
rows = [("Product", "Dedicated"), ("Product", "Class"), ("Product", "Shared"),
        ("Duration of stay", "Dedicated"), ("Duration of stay", "Class"), ("Duration of stay", "Shared")]
placements = ["Closest open (COL)", "Random", "Target"]

status = {
    ("Product", "Dedicated"): [("S", "Selected 1\n(mandatory)"), ("S", "Selected 2\n(random storage)"), ("S", "Selected 3\n(velocity slotting)")],
    ("Product", "Class"): [("S", "Selected 4"), ("S", "Selected 5"), ("S", "Selected 6")],
    ("Product", "Shared"): [("S", "Selected 7"), ("U", "Not supported"), ("S", "Selected 8")],
    ("Duration of stay", "Dedicated"): [("U", "Not supported")] * 3,
    ("Duration of stay", "Class"): [("U", "Not supported")] * 3,
    ("Duration of stay", "Shared"): [("R", "Rejected:\nduplicates 7"), ("U", "Not supported"), ("S", "Selected 9")],
}

fill_color = {"S": "#2563eb", "R": "#9ca3af", "U": "#f3f4f6"}
text_color = {"S": "white", "R": "white", "U": "#6b7280"}

fig, ax = plt.subplots(figsize=(8.2, 4.4))

for i, row in enumerate(rows):
    for j in range(len(placements)):
        kind, label = status[row][j]
        ax.add_patch(Rectangle((j + 0.03, -i - 0.97), 0.94, 0.94, fc=fill_color[kind], ec="white", lw=2,
                               hatch="///" if kind == "U" else None))
        ax.text(j + 0.5, -i - 0.5, label, ha="center", va="center", color=text_color[kind],
                fontsize=8.5, weight="bold" if kind == "S" else "normal")
    ax.text(-0.05, -i - 0.5, row[0] + " · " + row[1], ha="right", va="center")

for j, placement in enumerate(placements):
    ax.text(j + 0.5, 0.15, placement, ha="center", va="bottom", weight="bold")

ax.text(-0.05, 0.15, "Assigned by · Slots are", ha="right", va="bottom", weight="bold")
ax.axhline(-3, color="black", lw=0.8)
ax.set_xlim(-1.9, 3.02)
ax.set_ylim(-6.05, 0.6)
ax.axis("off")
ax.set_title("The 18 storage policy combinations: 9 selected, 8 not supported by the platform, 1 rejected",
             fontsize=10, loc="left")

plt.tight_layout()
plt.savefig("Output/T3a_fig_policy_matrix.png", dpi=170)
plt.show()

# figure 2: schematic of one temperature zone (6 x 10 = 60 slots) under each selected policy
rng = np.random.default_rng(7)

n_rows, n_cols = 6, 10
slots = [(r, c) for r in range(n_rows) for c in range(n_cols)]

# slots sorted from the pack bench (front middle) outwards
closest_first = sorted(slots, key=lambda s: (abs(s[1] - 4.5) + s[0] + 1, s[1]))

n_products = 44
velocity_rank = np.arange(n_products)                       # 0 = fastest mover
slots_needed = np.where(rng.random(n_products) < 0.18, 2, 1)
slots_needed[:4] = 2                                        # a few fast movers need two slots
product_number = rng.permutation(n_products)                # unrelated to velocity

velocity_class = np.where(velocity_rank < 9, "A", np.where(velocity_rank < 26, "B", "C"))
class_color = {"A": "#1e3a8a", "B": "#3b82f6", "C": "#bfdbfe"}

def assign(products, slot_order, need=None):
    """Give each product its slots, in order, from the slot list."""
    if need is None:
        need = slots_needed
    placed = {}
    k = 0
    for p in products:
        placed[p] = slot_order[k:k + need[p]]
        k += need[p]
    return placed

def draw_zone(ax, placed, title, subtitle, color_of=None):
    for s in slots:
        ax.add_patch(Rectangle((s[1], s[0]), 0.92, 0.92, fc="white", ec="#d1d5db", lw=0.8))
    for p, product_slots in placed.items():
        color = color_of(p) if color_of else class_color[velocity_class[p]]
        for s in product_slots:
            ax.add_patch(Rectangle((s[1], s[0]), 0.92, 0.92, fc=color, ec="white", lw=1))
    ax.add_patch(Rectangle((3.5, -1.25), 3.0, 0.75, fc="#fde68a", ec="#b45309", lw=1))
    ax.text(5.0, -0.88, "Pack bench", ha="center", va="center", fontsize=7)
    ax.set_xlim(-0.2, n_cols + 0.1)
    ax.set_ylim(-1.4, n_rows + 0.1)
    ax.axis("off")
    ax.set_title(title, fontsize=9, weight="bold", loc="left", pad=16)
    ax.text(0, n_rows + 0.25, subtitle, fontsize=7.3, color="#6b7280", va="bottom")

by_number = sorted(range(n_products), key=lambda p: product_number[p])
by_velocity = list(range(n_products))
random_slots = [slots[i] for i in rng.permutation(len(slots))]

# class areas: A nearest the bench, then B, then C
class_a = [p for p in range(n_products) if velocity_class[p] == "A"]
class_b = [p for p in range(n_products) if velocity_class[p] == "B"]
class_c = [p for p in range(n_products) if velocity_class[p] == "C"]
size_a = slots_needed[class_a].sum()
size_b = slots_needed[class_b].sum()
area_a = closest_first[:size_a + 1]
area_b = closest_first[size_a + 1:size_a + size_b + 2]
area_c = closest_first[size_a + size_b + 2:]

def class_assign(order_a, order_b, order_c, shuffle=False):
    placed = {}
    for products, area in [(order_a, area_a), (order_b, area_b), (order_c, area_c)]:
        if shuffle:
            area = [area[i] for i in rng.permutation(len(area))]
        placed.update(assign(products, area))
    return placed

# shared storage holds only today's stock, so some two-slot products need one slot today
need_today = np.where((slots_needed == 2) & (rng.random(n_products) < 0.6), 1, slots_needed)

# duration of stay: expected stay of today's stock, short stays nearest the bench
stay_rank = rng.permutation(n_products)
by_stay = sorted(range(n_products), key=lambda p: velocity_rank[p] * 0.6 + stay_rank[p] * 0.4)
stay_color = lambda p: ["#7c2d12", "#ea580c", "#fdba74"][
    0 if by_stay.index(p) < 12 else (1 if by_stay.index(p) < 28 else 2)]

fig, axes = plt.subplots(3, 3, figsize=(11, 8.6))

draw_zone(axes[0, 0], assign(by_number, closest_first), "1  Product · Dedicated · COL",
          "Fixed slots, given closest-first in product-number order")
draw_zone(axes[0, 1], assign(by_velocity, random_slots), "2  Product · Dedicated · Random",
          "Fixed slots at random locations")
draw_zone(axes[0, 2], assign(by_velocity, closest_first), "3  Product · Dedicated · Target",
          "Fixed slots, fastest movers nearest the bench")
draw_zone(axes[1, 0], class_assign(sorted(class_a, key=lambda p: product_number[p]),
                                   sorted(class_b, key=lambda p: product_number[p]),
                                   sorted(class_c, key=lambda p: product_number[p])),
          "4  Product · Class · COL", "A/B/C areas; closest open slot within each")
draw_zone(axes[1, 1], class_assign(class_a, class_b, class_c, shuffle=True), "5  Product · Class · Random",
          "A/B/C areas; random slot within each")
draw_zone(axes[1, 2], class_assign(class_a, class_b, class_c), "6  Product · Class · Target",
          "A/B/C areas; fastest first within each")
draw_zone(axes[2, 0], assign(by_number, closest_first, need_today), "7  Product · Shared · COL",
          "Any product, any open slot; filled closest-first as stock arrives")
draw_zone(axes[2, 1], assign(by_velocity, closest_first, need_today), "8  Product · Shared · Target",
          "Any open slot, aimed by velocity")
draw_zone(axes[2, 2], assign(by_stay, closest_first, need_today), "9  Duration of stay · Shared · Target",
          "Stock expected to leave soonest nearest the bench", color_of=stay_color)

legend = [Patch(fc=class_color["A"], label="A: fastest (80% of picks)"), Patch(fc=class_color["B"], label="B"),
          Patch(fc=class_color["C"], label="C: slowest"), Patch(fc="white", ec="#d1d5db", label="Empty slot"),
          Patch(fc="#7c2d12", label="Short stay (policy 9)"), Patch(fc="#ea580c", label="Medium stay"),
          Patch(fc="#fdba74", label="Long stay")]
fig.legend(handles=legend, loc="lower center", ncol=7, frameon=False, fontsize=8)
fig.suptitle("How each selected policy places stock in one temperature zone (schematic, 60 slots)",
             x=0.01, ha="left", fontsize=11, weight="bold")

plt.tight_layout(rect=(0, 0.04, 1, 0.97))
plt.savefig("Output/T3a_fig_policy_schematics.png", dpi=170)
plt.show()

# figure 3: pick concentration by temperature zone (share of order lines against share of products)
fig, ax = plt.subplots(figsize=(7, 4.2))
zone_color = {"Ambient": "#8b7d6b", "Chilled": "#3b82f6", "Frozen": "#6d28d9"}

for zone in zones:
    zone_skus = sku_df.loc[sku_df["temperature_zone"] == zone, "sku_id"]
    zone_lines = lines_per_sku[lines_per_sku.index.isin(zone_skus)].sort_values(ascending=False)
    share_products = np.arange(1, len(zone_lines) + 1) / len(zone_lines) * 100
    share_lines = zone_lines.cumsum().values / zone_lines.sum() * 100
    ax.plot(share_products, share_lines, color=zone_color[zone], lw=2,
            label=f"{zone} ({len(zone_lines):,} products)")

ax.axvline(20, color="grey", lw=0.8, ls="--")
ax.axhline(80, color="grey", lw=0.8, ls="--")
ax.set_xlabel("Share of products, most-picked first (%)")
ax.set_ylabel("Share of order lines (%)")
ax.set_title("Pick concentration by temperature zone", loc="left", fontsize=10)
ax.set_xlim(0, 100)
ax.set_ylim(0, 101)
ax.legend(loc="lower right", frameon=False)

plt.tight_layout()
plt.savefig("Output/T3a_fig_pick_concentration.png", dpi=170)
plt.show()

# velocity classes per zone used by the Class policies: A = 80% of lines, B = next 15%, C = last 5%
class_rows = []

for zone in zones:
    zone_skus = sku_df.loc[sku_df["temperature_zone"] == zone, "sku_id"]
    zone_lines = lines_per_sku[lines_per_sku.index.isin(zone_skus)].sort_values(ascending=False)
    zone_cum = zone_lines.cumsum() / zone_lines.sum()
    zone_class = np.where(zone_cum <= 0.80, "A", np.where(zone_cum <= 0.95, "B", "C"))
    class_rows.append({"zone": zone,
                       "A_products": int((zone_class == "A").sum()),
                       "B_products": int((zone_class == "B").sum()),
                       "C_products": int((zone_class == "C").sum())})
    
print(pd.DataFrame(class_rows))