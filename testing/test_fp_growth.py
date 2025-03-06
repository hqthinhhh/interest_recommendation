import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load dataset (Reduce rows using sampling)
df = pd.read_csv("SocialMediaUsersDataset.csv")  # Use 20K rows

# Convert DOB to AgeGroup
df["AgeGroup"] = pd.cut(df["DOB"].apply(lambda x: 2025 - int(str(x)[:4])),
                        bins=[0, 18, 30, 50, 100],
                        labels=["Teen", "Young Adult", "Adult", "Senior"])

# Keep only required columns
df = df[["AgeGroup", "Gender", "Interests"]]

# Convert Interests to list
df["Interests"] = df["Interests"].str.replace("'", "").str.split(", ")

# Convert categorical data to transactions
transactions = df.explode("Interests")

# One-hot encode AgeGroup, Gender, and Interests
one_hot = pd.get_dummies(transactions, columns=["AgeGroup", "Gender", "Interests"])

# Use FP-Growth to find frequent itemsets
frequent_itemsets = fpgrowth(one_hot, min_support=0.03, use_colnames=True)

# Generate association rules based on Lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Filter rules where antecedents are AgeGroup & Gender and consequents are Interests
rules = rules[rules["antecedents"].apply(lambda x: any(i.startswith("AgeGroup") or i.startswith("Gender") for i in x)) &
              rules["consequents"].apply(lambda x: any(i.startswith("Interests") for i in x))]

# Save final rules
rules.to_csv("interest_recommendation_rules.csv", index=False)
print(f"Final rules saved to 'interest_recommendation_rules.csv'. {len(rules)} rules generated.")
