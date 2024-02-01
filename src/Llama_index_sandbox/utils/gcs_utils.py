import os
import json
from google.cloud import secretmanager, firestore
from google.oauth2 import service_account


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


def get_firestore_client():
    """
    Creates a Firestore client. If running locally, uses credentials from a
    JSON file whose path is specified in the FIRESTORE_CREDENTIALS_PATH environment variable.
    Otherwise, it retrieves credentials from Google Secret Manager.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise EnvironmentError("GOOGLE_CLOUD_PROJECT environment variable is not set.")

    # Path to service account key
    credentials_path = os.environ.get("FIRESTORE_CREDENTIALS_PATH")

    if credentials_path and os.path.exists(credentials_path):
        # Use service account credentials from the file path
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
    else:
        # Fetch service account credentials from Secret Manager
        service_account_info = get_secret(project_id, 'firestore-service-account')
        credentials_info = json.loads(service_account_info)
        credentials = service_account.Credentials.from_service_account_info(credentials_info)

    firestore_client = firestore.Client(credentials=credentials, project=project_id)
    return firestore_client


def set_secrets_from_cloud():
    """
    Sets the necessary secrets as environment variables from Secret Manager.
    """
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
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
            os.environ[secret_name] = get_secret(project_id, secret_name)
