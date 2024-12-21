import random
import pandas as pd

# Ensure dataset is loaded
file_path = 'dataset/eclipse_platform.csv'  # Update to your file path
dataset = pd.read_csv(file_path)

# Check required columns exist
required_columns = ["Issue_id", "Title", "Description", "Duplicated_issue"]
for col in required_columns:
    if col not in dataset.columns:
        raise ValueError(f"Missing required column: {col}")

# Step 1: Create a dictionary mapping Issue_id to its Title and Description
issue_map = dataset.set_index("Issue_id")[["Title", "Description"]].to_dict("index")

# Step 2: Generate duplicate pairs
duplicate_pairs = []
for _, row in dataset.iterrows():
    if not pd.isna(row["Duplicated_issue"]):  # Check if the issue has a duplicate reference
        try:
            duplicate_id = int(row["Duplicated_issue"])  # Convert to int
            if duplicate_id in issue_map:  # Ensure the duplicate ID exists in the dataset
                duplicate_pairs.append({
                    "Bug1": f"{row['Title']} {row['Description']}",
                    "Bug2": f"{issue_map[duplicate_id]['Title']} {issue_map[duplicate_id]['Description']}",
                    "Label": 1
                })
        except (ValueError, KeyError) as e:
            print(f"Error processing duplicate pair: {e}")
            continue

# Step 3: Generate non-duplicate pairs
non_duplicate_pairs = []
non_duplicate_issues = dataset[dataset["Duplicated_issue"].isna()]["Issue_id"].tolist()

# Ensure we have enough non-duplicate issues to sample from
if len(non_duplicate_issues) < 2:
    raise ValueError("Not enough non-duplicate issues to generate pairs.")

for _ in range(len(duplicate_pairs)):
    try:
        id1, id2 = random.sample(non_duplicate_issues, 2)  # Sample two non-duplicate issues
        non_duplicate_pairs.append({
            "Bug1": f"{issue_map[id1]['Title']} {issue_map[id1]['Description']}",
            "Bug2": f"{issue_map[id2]['Title']} {issue_map[id2]['Description']}",
            "Label": 0
        })
    except KeyError as e:
        print(f"Error processing non-duplicate pair: {e}")
        continue

# Combine both sets of pairs
pairs = duplicate_pairs + non_duplicate_pairs
pairs_df = pd.DataFrame(pairs)

# Save the processed pairs for future use
output_file = "processed_pairs.csv"
pairs_df.to_csv(output_file, index=False)
print(f"Duplicate and non-duplicate pairs generated and saved to {output_file}.")
