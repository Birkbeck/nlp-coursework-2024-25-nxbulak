import pandas as pd # sort and organise data

from pathlib import Path # to access files in other directories

def read_csv(csv_path=Path.cwd() / "p2-texts" / "hansard40000.csv"): # 2a) Read the handsard40000.csv dataset in the texts directory into a dataframe. Sub-set and rename the dataframe as follows:
    # check path works
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist!")
        return None
    # read csv & prints for testing
    #print(f"Reading csv in: {csv_path}")
    df = pd.read_csv(csv_path)
    #print(f"Dataframe pre adjustment: {df.shape}")

    # 2a) i. rename the 'Labour (Co-op)' value in party column to 'Labour', and then:
    df['party'] = df ['party'].replace('Labour (Co-op)', 'Labour')

    # 2a) ii. remove any rows where the value of the 'party' column is not one of the four most common party names, and remove the 'Speaker' value.
    df = df[df['party'] != 'Speaker']
    most_common_parties = df['party'].value_counts().head(4).index.tolist()
    df = df[df['party'].isin(most_common_parties)]

    print(df.shape)
    return df # show original df


if __name__ == "__main__":
    #Testing for 2a) - PASSED
    print("Testing 2a")
    print("-" * 30)

    df = read_csv()

    if df is not None:
        print("csv loaded")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nUnique parties in df: {df['party'].unique()}") # check party names have been changed
    else:
        print("Error: could not load")