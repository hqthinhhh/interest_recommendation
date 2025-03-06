import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv("dataset/SocialMediaUsersDataset.csv")

# Convert DOB to Age
df["Age"] = 2025 - pd.to_datetime(df["DOB"]).dt.year

print("1", df["Age"])

# Categorize age into groups
def categorize_age(age):
    if age < 18:
        return "Teenager"
    elif age < 30:
        return "Young Adult"
    elif age < 50:
        return "Adult"
    else:
        return "Senior"

df["Age Group"] = df["Age"].apply(categorize_age)
print("2", df["Age Group"])

# Convert Interests to lists
df["Interests"] = df["Interests"].apply(lambda x: eval(x) if isinstance(x, str) else [])
print("3", df["Interests"])

# **Fix: Use a list instead of .append()**
transactions = []

for _, row in df.iterrows():
    transaction = set(row["Interests"])  # Add Interests
    transaction.add(row["Gender"])       # Add Gender
    transaction.add(row["Age Group"])    # Add Age Group
    transaction.add(row["Country"])      # Add Country
    transaction.add(row["City"])         # Add City
    transactions.append(list(transaction))

print("4for loop", transactions)

# Create DataFrame from transactions
encoded_df = pd.DataFrame(transactions)

print("5")

# Convert to one-hot encoded format
one_hot = encoded_df.stack().str.get_dummies().groupby(level=0).sum()
print("6")

# Apply Apriori
frequent_itemsets = apriori(one_hot, min_support=0.15, use_colnames=True)
print("7")

# Generate rules based on Lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
print("8")

# Save results
rules.to_csv("content_recommendation_rules.csv", index=False)
print("9")

print(f"Generated {len(rules)} rules. Saved to 'content_recommendation_rules.csv'.")
