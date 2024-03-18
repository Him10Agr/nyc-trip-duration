import pandas as pd
import pathlib
import sys

def load_csv(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def save_csv(train_df: pd.DataFrame, test_df: pd.DataFrame, file_path: str):
    train_df.to_csv(file_path + '/train.csv', index=False)
    test_df.to_csv(file_path + '/test.csv', index=False)
    
if __name__ == '__main__':
    
    cur_dir = pathlib.Path(__file__)
    home_dir = cur_dir.parent.parent.parent
    #print(home_dir)
    sys.path.append(home_dir.as_posix())
    #print(sys.path)
    from src.features.build_features import feature_build, feature_drop
    
    input_path = sys.argv[1]    #'/data/raw'
    data_path = home_dir.as_posix() + input_path
    
    output_path = home_dir.as_posix() + '/data/processed'
    
    train_df = load_csv(data_path + '/train.csv')
    test_df = load_csv(data_path + '/test.csv')
    
    train_data = feature_build(train_df)
    test_data = feature_build(test_df)
    
    feature_drop_list_train = ['id', 'pickup_datetime', 'dropoff_datetime']
    train_data = feature_drop(df = train_data, feature_drop_list= feature_drop_list_train)
    feature_drop_list_test = ['id', 'pickup_datetime']
    test_data = feature_drop(df = test_data, feature_drop_list= feature_drop_list_test)
    
    save_csv(train_df=train_data, test_df = test_data, file_path=output_path)
    
    