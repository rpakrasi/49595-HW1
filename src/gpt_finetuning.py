import os
from openai import AzureOpenAI
import keys

# Load the OpenAI client
client = AzureOpenAI(
    azure_endpoint=keys.azure_openai_endpoint,
    api_key=keys.azure_openai_key,
    api_version=keys.azure_openai_api_version
)

# Upload the training and validation dataset files to Azure OpenAI
training_file_name = 'training_set.jsonl'
validation_file_name = 'validation_set.jsonl'

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

# Create a fine-tuning job
response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-4o-mini-2024-07-18", # Enter the base model name
    suffix="my-model", # Custom suffix for naming the resulting model
    seed=105, # Seed parameter controls reproducibility of the fine-tuning job
    extra_body={"trainingType": "GlobalStandard"} # Change to Standard or Developer as needed
)

job_id = response.id

# You can use the job ID to monitor the status of the fine-tuning job
# The fine-tuning job takes some time to start and finish

print("Job ID:", response.id)
print(response.model_dump_json(indent=2))

# Check the status of the fine-tuning job
response = client.fine_tuning.jobs.retrieve(job_id)

print("Job ID:", response.id)
print("Status:", response.status)
print(response.model_dump_json(indent=2))

# List fine-tuning events (optional)
response = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10)
print(response.model_dump_json(indent=2))

# Download results after job completes (optional)
# Uncomment after the job succeeds
# response = client.fine_tuning.jobs.retrieve(job_id)
# if response.status == 'succeeded':
#     result_file_id = response.result_files[0]
#     retrieve = client.files.retrieve(result_file_id)
#     
#     print(f'Downloading result file: {result_file_id}')
#     with open(retrieve.filename, "wb") as file:
#         result = client.files.content(result_file_id).read()
#         file.write(result)
