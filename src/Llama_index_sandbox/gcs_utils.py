import os
from google.cloud import secretmanager


def set_secrets_from_cloud():
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project_id:
        client = secretmanager.SecretManagerServiceClient()
        secrets_to_fetch = {
            'OPENAI_API_KEY': 'OPENAI_API_KEY',
            'OPENAI_API_KEY_ANYSCALE': 'OPENAI_API_KEY_ANYSCALE',
            'OPENAI_API_BASE_ANYSCALE': 'OPENAI_API_BASE_ANYSCALE',
            'ASSEMBLY_AI_API_KEY': 'ASSEMBLY_AI_API_KEY',
            'PINECONE_API_KEY': 'PINECONE_API_KEY',
            'PINECONE_API_ENVIRONMENT': 'PINECONE_API_ENVIRONMENT',
            'ASSEMBLY_AI_API': 'ASSEMBLY_AI_API',
            'YOUTUBE_CHANNELS': 'YOUTUBE_CHANNELS',
            'YOUTUBE_API_KEY': 'YOUTUBE_API_KEY',
        }

        for secret_name, env_name in secrets_to_fetch.items():
            if env_name not in os.environ:
                secret_version = client.access_secret_version(
                    request={
                        "name": f"projects/{project_id}/secrets/{secret_name}/versions/latest"
                    }
                )
                secret_value = secret_version.payload.data.decode("UTF-8")
                os.environ[env_name] = secret_value


# Call the function at the beginning of your application

