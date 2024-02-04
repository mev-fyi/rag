import logging
import os
from google.cloud import secretmanager


def get_secret(project_id, secret_name):
    """
    Retrieves a secret from Google Cloud Secret Manager.
    """
    if not project_id:
        raise ValueError("Google Cloud project ID is not set.")

    client = secretmanager.SecretManagerServiceClient()
    secret_version_name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": secret_version_name})
    secret_data = response.payload.data.decode("UTF-8")
    return secret_data


def set_secrets_from_cloud():
    """
    Sets the necessary secrets as environment variables from Secret Manager.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "mev-fyi")
    secrets_to_fetch = [
        'OPENAI_API_KEY',
        'ASSEMBLY_AI_API_KEY',
        'PINECONE_API_KEY',
        'PINECONE_API_ENVIRONMENT',
        'ASSEMBLY_AI_API',
        'YOUTUBE_CHANNELS',
        'YOUTUBE_API_KEY',
        'TWITTER_CONSUMER_KEY',
        'TWITTER_CONSUMER_SECRET',
        'TWITTER_ACCESS_TOKEN',
        'TWITTER_ACCESS_TOKEN_SECRET',
        'TWITTER_CLIENT_ID',
        'TWITTER_CLIENT_SECRET',
        'TWITTER_BEARER_TOKEN',
        'TWITTER_BOT',
        'TWITTER_USERNAME',
        'TWITTER_PASSWORD',
        'NEXTJS_API_ENDPOINT',
        'NEXTJS_API_KEY',
    ]

    for secret_name in secrets_to_fetch:
        if secret_name not in os.environ:
            logging.info(f"set_secrets_from_cloud: fetching [{secret_name}] from Secrets Manager")
            os.environ[secret_name] = get_secret(project_id, secret_name)
