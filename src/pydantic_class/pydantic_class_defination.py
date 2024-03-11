import pathlib
import pandas as pd


def pydantic_class(test_data_path):
    
    import pandas as pd
    
    df_columns_dtypes = pd.read_csv(test_data_path).dtypes.to_frame().reset_index()

    class_def = f'''from pydantic import BaseModel\n\nclass PredictionInput(BaseModel):\n'''
    for i in range(1,df_columns_dtypes.shape[0]):
        
        if str(df_columns_dtypes[0][i]) == 'int64':
            class_def = class_def + f'''\t{df_columns_dtypes['index'][i]}''' + ':' + f''' int\n'''
        if str(df_columns_dtypes[0][i]) == 'float64':
            class_def = class_def + f'''\t{df_columns_dtypes['index'][i]}''' + ':' + f''' float\n'''
        if str(df_columns_dtypes[0][i]) == 'object':
            class_def = class_def + f'''\t{df_columns_dtypes['index'][i]}''' + ':' + f''' object\n'''


    with open('src/pydantic_class/pydantic_class.py', 'w') as file:
        file.write(class_def)


        
    