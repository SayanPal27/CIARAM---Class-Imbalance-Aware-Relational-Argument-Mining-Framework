import pandas as pd
import json
import random

# Load CSV file
df = pd.read_csv('debate.csv')

# Filter relevant columns
df_filtered = df[['Arg1', 'Arg2', 'rel', 'split']]

# Generate random 8-digit unique IDs
df_filtered['id'] = [random.randint(10000000, 99999999) for _ in range(len(df_filtered))]

# Map 0 to "Attack" and 1 to "Support" in the 'rel' column
df_filtered['rel'] = df_filtered['rel'].map({0: "Attack", 1: "Support"})

# Split the data into train, validation, and test sets based on the 'split' column
train_data = df_filtered[df_filtered['split'] == 'train']
val_data = df_filtered[df_filtered['split'] == 'dev']
test_data = df_filtered[df_filtered['split'] == 'test']

# Convert to JSON format
def convert_to_json(dataframe):
    return dataframe[['id', 'Arg1', 'Arg2', 'rel']].to_dict(orient='records')

train_json = convert_to_json(train_data)
val_json = convert_to_json(val_data)
test_json = convert_to_json(test_data)

# Save each set to separate JSON files
with open('debate_train.json', 'w') as train_file:
    json.dump(train_json, train_file, indent=4)

with open('debate_dev.json', 'w') as val_file:
    json.dump(val_json, val_file, indent=4)

with open('debate_test.json', 'w') as test_file:
    json.dump(test_json, test_file, indent=4)

print("Files successfully created!")
