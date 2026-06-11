import os
from functools import lru_cache
from dotenv import load_dotenv

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference, TSModelInference

load_dotenv()


@lru_cache(maxsize=1)
def get_client() -> APIClient:
    api_key    = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    url        = os.getenv("WATSONX_URL")

    if not all([api_key, project_id, url]):
        raise EnvironmentError("Variables manquantes dans .env : WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL")

    assert api_key and project_id and url
    client = APIClient(Credentials(url=url, api_key=api_key))
    client.set.default_project(project_id)
    return client


def get_llm(model_id: str = "meta-llama/llama-3-3-70b-instruct") -> ModelInference:
    return ModelInference(model_id=model_id, api_client=get_client())


def get_ts_model(model_id: str = "ibm/granite-ttm-512-96-r2") -> TSModelInference:
    """
    Modèles disponibles :
      - ibm/granite-ttm-512-96-r2   (context=512,  prediction=96)
      - ibm/granite-ttm-1024-96-r2  (context=1024, prediction=96)
      - ibm/granite-ttm-1536-96-r2  (context=1536, prediction=96)
    """
    return TSModelInference(model_id=model_id, api_client=get_client())
