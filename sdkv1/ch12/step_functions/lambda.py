import boto3, time

def lambda_handler(event, context):
    print(event)
    endpoint_name = event['EndpointArn'].split('/')[-1]
    sm = boto3.client('sagemaker')
    waiter = sm.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    return {
        'statusCode': 200,
        'body': endpoint_name
    }
