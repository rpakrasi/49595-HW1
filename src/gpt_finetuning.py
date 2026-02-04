import time

from openai import AzureOpenAI

import keys

# Load the OpenAI client
client = AzureOpenAI(
    azure_endpoint=keys.azure_openai_endpoint,
    api_key=keys.azure_openai_key,
    api_version=keys.azure_openai_api_version
)

# Upload the training and validation dataset files to Azure OpenAI
training_file_name = '../data/trump-responses/training.jsonl'
validation_file_name = '../data/trump-responses/validation.jsonl'

training_response = client.files.create(
    file=open(training_file_name, "rb"),
    purpose="fine-tune"
)
validation_response = client.files.create(
    file=open(validation_file_name, "rb"),
    purpose="fine-tune"
)
training_file_id = training_response.id
validation_file_id = validation_response.id

print("Training file ID:", training_file_id)
print("Validation file ID:", validation_file_id)


def wait_for_file(file_id):
    while True:
        f = client.files.retrieve(file_id)
        print(f"File {file_id} status: {f.status}")
        if f.status == "processed":
            break
        elif f.status == "error":
            raise Exception(f"File failed processing: {f.status_details}")
        time.sleep(2)


wait_for_file(training_file_id)
wait_for_file(validation_file_id)

# Create a fine-tuning job
response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-4o-2024-08-06",  # Enter the base model name
    suffix="nlp-hw1",  # Custom suffix for naming the resulting model
    seed=105,  # Seed parameter controls reproducibility of the fine-tuning job
    extra_body={"trainingType": "Developer"}  # Change to Standard or GlobalStandard as needed
)

job_id = response.id

# # You can use the job ID to monitor the status of the fine-tuning job
# # The fine-tuning job takes some time to start and finish

print("Job ID:", response.id)
print(response.model_dump_json(indent=2))
