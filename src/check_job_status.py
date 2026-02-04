from openai import AzureOpenAI

import keys

job_id = 'ftjob-399186892c10464e916e6016810ba695'
client = AzureOpenAI(
    azure_endpoint=keys.azure_openai_endpoint,
    api_key=keys.azure_openai_key,
    api_version=keys.azure_openai_api_version
)

response = client.fine_tuning.jobs.retrieve(job_id)

print("Job ID:", job_id)
print("Response/Job ID:", response.id)
print("Status:", response.status)
print(response.model_dump_json(indent=2))

if response.status == 'succeeded':
    result_file_id = response.result_files[0]
    retrieve = client.files.retrieve(result_file_id)

    print(f'Downloading result file: {result_file_id}')
    with open(retrieve.filename, "wb") as file:
        result = client.files.content(result_file_id).read()
        file.write(result)
