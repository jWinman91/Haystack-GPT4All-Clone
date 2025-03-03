from pydantic import BaseModel
from typing import Optional


class QueryParams(BaseModel):
    llm_model_name: str
    embedding_model_name: str
    prompt_template: str
