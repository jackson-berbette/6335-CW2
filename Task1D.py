import pandas as pd
from itertools import combinations
from collections import Counter
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

order_lines_df = pd.read_csv("Data/QF_Facility_Order_Lines.csv")

# get just the order and skus
order_skus = order_lines_df[["order_id", "sku_id"]]

# organize each order into a list of skus
baskets = order_skus.groupby("order_id")["sku_id"].apply(list)

# get number of orders
num_orders = len(baskets)

# get number of unique skus
sku_counts = Counter(sku for basket in baskets for sku in basket)

# get number of unique sku pairs
pair_counts = Counter()
for basket in baskets:
    for a, b in combinations(sorted(basket), 2):
        pair_counts[(a, b)] += 1

# we will calculate support, lift, and significance
results = []

# pair must occur in at least 200 orders (arbitrary)
min_orders_pair = 200

# first product in a pair is "a", second is "b"
for (a,b), num_ab in pair_counts.items():
    if num_ab < min_orders_pair:
        continue
    
    num_a = sku_counts[a]
    num_b = sku_counts[b]

    # support = number of orders containing current pair / total number of orders
    support = num_ab / num_orders

    confidence_ab = num_ab / num_a
    confidence_ba = num_ab / num_b

    # expected co-occurence under independence: num_a * num_b / num_orders
    expected = num_a * num_b / num_orders

    # calculate lift -- "how much more often the pair occurs than expected if independent"
    # Lift = 1 --> no association, Lift > 1 --> positive association, Lift < 1 --> negative
    lift = num_ab / expected

    # Fisher's test
    p_value = hypergeom.sf(num_ab-1, num_orders, num_a, num_b)

    results.append({
        "sku_a": a,
        "sku_b": b,
        "num_orders_together": num_ab,
        "support": support,
        "confidence_ab": confidence_ab,
        "confidence_ba": confidence_ba,
        "expected_orders": expected,
        "lift": lift,
        "p_value": p_value
    })

pairs_df = pd.DataFrame(results)

# correct for testing many pairs -- Benjamini-Hochberg
if not pairs_df.empty:
    pairs_df["adjusted_p_value"] = multipletests(
        pairs_df["p_value"], method="fdr_bh"
    )[1]

    pairs_df = pairs_df.sort_values(
        ["lift"], ascending=False
    )

# record statistically significant pairs with at least 25% lift
significant_pairs = (
    pairs_df[
        (pairs_df["adjusted_p_value"] < 0.05) &
        (pairs_df["lift"] >= 1.25)
    ][
        ["sku_a", "sku_b", "num_orders_together", "lift"]
    ]
    .sort_values("lift", ascending=False)
    .reset_index(drop=True)
)

print(significant_pairs)
