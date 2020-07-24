import boto3
from load_dataset import load_dataset

app_name = 'mlflow-xgb-demo'
region = 'eu-west-1'

if __name__ == '__main__':
    sm = boto3.client('sagemaker', region_name=region)
    smrt = boto3.client('runtime.sagemaker', region_name=region)

    # Check endpoint status
    endpoint = sm.describe_endpoint(EndpointName=app_name)
    print("Endpoint status: ", endpoint["EndpointStatus"])
    # Load test set
    x_train, x_test, y_train, y_test = load_dataset(
        'bank-additional/bank-additional-full.csv', ';'
    )
    # Predict first 10 samples
    input_data = x_test[:10].to_json(orient="split")
    prediction = smrt.invoke_endpoint(
        EndpointName=app_name,
        Body=input_data,
        ContentType='application/json; format=pandas-split'
    )
    prediction = prediction['Body'].read().decode("ascii")
    print(prediction)
