import os

from dotenv import load_dotenv

load_dotenv()

azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = "https://siskind-openai.openai.azure.com/"
# "openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview" - was originally part of the endpoint url
azure_openai_region = "East US 2"
azure_openai_api_version = "2024-08-01-preview"

azure_key = os.getenv("AZURE_KEY")
azure_endpoint = "https://eastus.api.cognitive.microsoft.com/"
azure_region = "eastus"
