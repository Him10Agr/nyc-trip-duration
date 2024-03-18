import pathlib
import pandas as pd

data = pathlib.Path(__file__).parent.as_posix() + '/data/raw/test.csv'

test_df = pd.read_csv(data)

print(test_df.sample(5))