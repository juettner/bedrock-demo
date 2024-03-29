import boto3
import botocore.config
import requests
import json

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-west-2',
    config=botocore.config.Config(
        read_timeout=300
    )
)

def lambda_handler(event, context):

    prompt = event['prompt']
    temperature = int(event['temperature'])
    topP = float(event['topP'])
    model = event['model']

    body = json.dumps({
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 128,
            "stopSequences": [],
            "temperature": temperature,
            "topP": topP
        }
    })

    try:
        response = bedrock.invoke_model(
            body=body,
            modelId=model,
            accept='application/json',
            contentType='application/json'
        )

        response_content = response.get('body').read().decode('utf-8')
        response_data = json.loads(response_content)

    except requests.RequestException as e:
        print(e)
        raise e

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": response_data
        }),
    }
