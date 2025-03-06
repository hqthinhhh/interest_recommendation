import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load dataset (Adjust the file name as needed)
df = pd.read_csv("dataset/SocialMediaUsersDataset.csv")

# Keep only required columns
df = df[["Gender", "Interests"]]

# Convert Interests to a list, ensuring proper formatting
df["Interests"] = df["Interests"].str.replace("'", "").str.split(", ")

# Convert categorical data to transactions
transactions = df.explode("Interests")

# One-hot encode Gender and Interests
one_hot = pd.get_dummies(transactions, columns=["Gender", "Interests"], dtype=int)

# Aggregate transactions by summing one-hot encoded values
one_hot = one_hot.groupby(one_hot.index).max()

# Use FP-Growth to find frequent itemsets
frequent_itemsets = fpgrowth(one_hot, min_support=0.01, use_colnames=True)

# Generate association rules based on Lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)

# Filter rules where antecedents are Gender and consequents are Interests
rules = rules[
    rules["antecedents"].apply(lambda x: any(i.startswith("Gender_") for i in x)) &
    rules["consequents"].apply(lambda x: any(i.startswith("Interests_") for i in x))
]

# Function to clean prefixes
def clean_prefix(itemset):
    return ", ".join(sorted(i.replace("Gender_", "").replace("Interests_", "") for i in itemset))

# Apply cleaning to antecedents and consequents
rules["antecedents"] = rules["antecedents"].apply(clean_prefix)
rules["consequents"] = rules["consequents"].apply(clean_prefix)

rules = rules[["antecedents","consequents","antecedent support","consequent support","support","confidence","lift"]]

# Save final rules
rules.to_csv("results/gender_interest_recommendation_rules.csv", index=False)
print(f"Final rules saved to 'gender_interest_recommendation_rules.csv'. {len(rules)} rules generated.")
