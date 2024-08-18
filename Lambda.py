"""
 serializeImageData:  Lambda function to serialize the image data
"""

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
   
    # Get the s3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from s3 to /tmp/image.png
    boto3.resource('s3').Bucket(bucket).download_file(key, "/tmp/image.png")
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }



"""
ImageClassifier : Lambda function to predict image classification
"""

iimport os
import io
import boto3
import json
# import sagemaker
import base64
# from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-08-14-17-34-36-614"

# Initialize the SageMaker runtime client
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    # # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])
    
    # Make a prediction:
    predictor = runtime.invoke_endpoint(EndpointName=ENDPOINT,
                                    #   ContentType='image/png',
                                    ContentType='application/x-image',
                                      Body=image)
    
    # We return the data back to the Step Function    
    event["inferences"] = json.loads(predictor['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,

        "body": {
            "image_data": event["body"]['image_data'],
            "s3_bucket": event["body"]['s3_bucket'],
            "s3_key": event["body"]['s3_key'],
            "inferences": event['inferences'],
       }
}




"""
InferenceeFilter : Lambda function tofiter inference 
"""

import json


THRESHOLD = .90


def lambda_handler(event, context):
    # Get the inferences from the event
    inferences = event["body"]["inferences"]
    
    # Check if any values in any inferences are below THRESHOLD
    meets_threshold = (max(inferences) > THRESHOLD)
    
    # If the threshold is not met, raise an exception to fail the step
    if not meets_threshold:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    # If the threshold is met, pass the data back out of the Step Function
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }