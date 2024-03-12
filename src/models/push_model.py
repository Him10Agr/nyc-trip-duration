import os
import boto3
import shutil
from botocore.exceptions import NoCredentialsError

os.environ["AWS_ACCESS_KEY_ID"] = 'AKIAX4573WIYKDPXD2SB'
os.environ["AWS_SECRET_ACCESS_KEY"] = '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'
#os.environ["AWS_DEFAULT_REGION"] = 'ap-south-1'


def upload_to_s3(local_file_path, bucket_name, s3_file_path):
    
    s3 = boto3.client('s3')
    
    try:
        s3.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f'File Successfully loaded to {bucket_name}/{s3_file_path}')
    except FileNotFoundError:
        print(f'The file {local_file_path} was not found')
    except NoCredentialsError:
        print('Credential not available')
        
local_file_path = 'models/model.pkl'
s3_bucket_name = 'nyc-taxi-app'
s3_file_path = 'models/model.pkl'

upload_to_s3(local_file_path, s3_bucket_name, s3_file_path)

'''docker not able to read credential of aws and 
able to pull model from s3 to form image in CI/CD therefore model.pkl 
file copied outside to push in git'''

shutil.copy(local_file_path, 'model.pkl')