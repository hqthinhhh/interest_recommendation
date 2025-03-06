import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load dataset
df = pd.read_csv("dataset/SocialMediaUsersDataset.csv")

# Convert DOB to AgeGroup
df["AgeGroup"] = pd.cut(df["DOB"].apply(lambda x: 2025 - int(str(x)[:4])),
                        bins=[0, 18, 30, 50, 100],
                        labels=["Teen", "Young Adult", "Adult", "Senior"])

# Keep only required columns
df = df[["AgeGroup", "Gender", "Interests"]]

# Convert Interests to list
df["Interests"] = df["Interests"].str.replace("'", "").str.split(", ")

# Explode interests into separate rows
df = df.explode("Interests")

# One-hot encode Gender, AgeGroup, and Interests
one_hot = pd.get_dummies(df, columns=["Gender", "AgeGroup", "Interests"])

# Use FP-Growth to find frequent itemsets
frequent_itemsets = fpgrowth(one_hot, min_support=0.01, use_colnames=True)

# Generate association rules based on Lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)

# Filter rules where antecedents are (Gender, AgeGroup) and consequents are Interests
rules = rules[rules["antecedents"].apply(lambda x: any(i.startswith("Gender") or i.startswith("AgeGroup") for i in x)) &
              rules["consequents"].apply(lambda x: any(i.startswith("Interests") for i in x))]

# Normalize column values by removing prefixes
def normalize_frozenset(fset):
    return ", ".join(sorted([i.replace("Gender_", "").replace("AgeGroup_", "").replace("Interests_", "") for i in fset]))

rules["antecedents"] = rules["antecedents"].apply(normalize_frozenset)
rules["consequents"] = rules["consequents"].apply(normalize_frozenset)

# Save final rules
rules.to_csv("results/gender_and_age_interest_recommendation_rules.csv", index=False)
print(f"Final rules saved to 'interest_recommendation_rules.csv'. {len(rules)} rules generated.")
