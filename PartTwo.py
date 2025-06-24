import pandas as pd # sort and organise data

from pathlib import Path # to access files in other directories

def read_csv(csv_path=Path.cwd() / "p2-texts" / "hansard40000.csv", verbose=False): # 2a) Read the handsard40000.csv dataset in the texts directory into a dataframe. Sub-set and rename the dataframe as follows:
    # check path works
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist!")
        return None

    # read csv
    print(f"Reading csv in: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataframe pre adjustment: {df.shape}")

    return df # show original df


if __name__ == "__main__":
    # Testing for 2a)
    print("Testing 2a")
    print("-" * 30)

    df = read_csv()

    if df is not None:
        print("csv loaded")
        print(f"Columns: {df.columns.tolist()}")
    else:
        print("Error: could not load")