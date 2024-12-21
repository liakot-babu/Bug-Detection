import re
import pandas as pd

# Load the dataset
file_path = 'processed_pairs.csv'  # Update to your file path
pairs_df = pd.read_csv(file_path)

# Ensure necessary columns exist
required_columns = ["Bug1", "Bug2"]
for col in required_columns:
    if col not in pairs_df.columns:
        raise ValueError(f"Missing required column: {col}")

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""  # Return empty string for invalid or missing data
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip().lower()  # Convert to lowercase and strip whitespace

# Apply cleaning to Bug1 and Bug2 columns
pairs_df["Bug1"] = pairs_df["Bug1"].apply(clean_text)
pairs_df["Bug2"] = pairs_df["Bug2"].apply(clean_text)

# Save the cleaned data
output_file = "cleaned_pairs.csv"
pairs_df.to_csv(output_file, index=False)
print(f"Text cleaned and saved to {output_file}.")
