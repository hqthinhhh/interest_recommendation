import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load dataset
df = pd.read_csv("dataset/SocialMediaUsersDataset.csv")

# Convert DOB to Age
df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")  # Convert DOB to datetime
df["Age"] = (pd.Timestamp.today().year - df["DOB"].dt.year).fillna(0).astype(int)  # Calculate Age

# Define age groups
def categorize_age(age):
    if age < 18:
        return "Under_18"
    elif 18 <= age < 25:
        return "18_24"
    elif 25 <= age < 35:
        return "25_34"
    elif 35 <= age < 45:
        return "35_44"
    elif 45 <= age < 55:
        return "45_54"
    elif 55 <= age < 65:
        return "55_64"
    else:
        return "65_plus"

# Apply age categorization
df["Age Group"] = df["Age"].apply(categorize_age)

# Keep only relevant columns
df = df[["Age Group", "Gender", "Interests"]]

# Convert Interests to a list, ensuring proper formatting
df["Interests"] = df["Interests"].astype(str).str.replace("'", "").str.split(", ")

# Create a combined "Age Group + Gender" column
df["Age_Gender"] = df["Age Group"] + "_" + df["Gender"].astype(str)

# Convert categorical data to transactions
transactions = df.explode("Interests")

# One-hot encode Age_Gender and Interests
one_hot = pd.get_dummies(transactions, columns=["Age_Gender", "Interests"], dtype=int)

# Aggregate transactions by summing one-hot encoded values
one_hot = one_hot.groupby(one_hot.index).max()

# Ensure all values are numeric and binary
one_hot = one_hot.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# Use FP-Growth to find frequent itemsets
frequent_itemsets = fpgrowth(one_hot, min_support=0.01, use_colnames=True)

# Generate association rules based on Lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.01)

# Filter rules where antecedents are Age_Gender and consequents are Interests
rules = rules[
    rules["antecedents"].apply(lambda x: any(i.startswith("Age_Gender_") for i in x)) &
    rules["consequents"].apply(lambda x: any(i.startswith("Interests_") for i in x))
]

# Function to clean prefixes
def clean_prefix(itemset):
    return ", ".join(sorted(i.replace("Age_Gender_", "").replace("Interests_", "") for i in itemset))

# Apply cleaning to antecedents and consequents
rules["antecedents"] = rules["antecedents"].apply(clean_prefix)
rules["consequents"] = rules["consequents"].apply(clean_prefix)

# Keep only necessary columns
rules = rules[["antecedents","consequents","antecedent support","consequent support","support","confidence","lift"]]

# Save final rules
rules.to_csv("results/age_gender_interest_recommendation_rules.csv", index=False)
print(f"Final rules saved to 'age_gender_interest_recommendation_rules.csv'. {len(rules)} rules generated.")