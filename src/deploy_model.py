import json
import os

import requests
from openai import AzureOpenAI

import keys

# TODO before running:
#  Ask for subscription ID + resource group + azure openai resource name + token.
#  Ask for permission to deploy model (costs)

client = AzureOpenAI(
    azure_endpoint=keys.azure_openai_endpoint,
    api_key=keys.azure_openai_key,
    api_version=keys.azure_openai_api_version
)

job = client.fine_tuning.jobs.retrieve("ftjob-399186892c10464e916e6016810ba695")

FINE_TUNED_MODEL_ID = job.fine_tuned_model

token = os.getenv("<TOKEN>")
subscription = "<YOUR_SUBSCRIPTION_ID>"
resource_group = "<YOUR_RESOURCE_GROUP_NAME>"
resource_name = "<YOUR_AZURE_OPENAI_RESOURCE_NAME>"
model_deployment_name = "gpt-41-ft"  # Custom deployment name that you use to reference the model when making inference calls.

deploy_params = {'api-version': "2024-10-01"}  # Control plane API version rather than the data plane API for this call
deploy_headers = {'Authorization': 'Bearer {}'.format(token), 'Content-Type': 'application/json'}

deploy_data = {
    "sku": {"name": "standard", "capacity": 1},
    "properties": {
        "model": {
            "format": "OpenAI",
            "name": FINE_TUNED_MODEL_ID,
            # Retrieve this value from the previous call; it looks like gpt-4.1-2025-04-14.ft-b044a9d3cf9c4228b5d393567f693b83
            "version": "1"
        }
    }
}
deploy_data = json.dumps(deploy_data)

request_url = f'https://management.azure.com/subscriptions/{subscription}/resourceGroups/{resource_group}/providers/Microsoft.CognitiveServices/accounts/{resource_name}/deployments/{model_deployment_name}'

print('Creating a new deployment...')

r = requests.put(request_url, params=deploy_params, headers=deploy_headers, data=deploy_data)

print(r)
print(r.reason)
print(r.json())
