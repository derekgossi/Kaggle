import pandas as pd

# Read training data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

## Data preprocessing

# Integer feature names
train_df.columns = ["id"] + range(1,94) + ["target"]
test_df.columns = ["id"] + range(1,94)

# Add train and test column
train_df["test"] = 0
test_df["test"] = 1

# Change target value to integer class
for old_class in train_df["target"].unique():
    new_class = list(old_class)[-1:]
    new_class = int(new_class[0])
    train_df["target"] = train_df["target"].replace(old_class, new_class)

# Merge two dataframes
df = pd.concat([train_df, test_df])

# Pickle dataframe
import pickle

with open('pickle/processed_df.p', mode='wb') as picklefile:
	pickle.dump(df, picklefile)
