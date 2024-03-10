import pandas as pd
import pathlib
import sys

def load_csv(file_path) -> pd.DataFrame:
    df = load_csv(file_path)
    return df

def save_csv(train_df: pd.DataFrame, test_df: pd.DataFrame, file_path: str):
    train_df.to_csv(file_path + 'train.csv')
    test_df.to_csv(file_path + 'test.csv')
    
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
    
    save_csv(train_df=train_data, test_df = test_df, file_path=output_path)
    
    