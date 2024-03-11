import lazypredict
from lazypredict.Supervised import LazyRegressor
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
train_data_path = home_dir.as_posix() + '/data/processed/train.csv'
train_data = pd.read_csv(train_data_path)

X = train_data.drop(columns=['trip_duration'], axis=1)
y = train_data['trip_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models, prediction = reg.fit(X_train, X_test, y_train, y_test)

print(models)