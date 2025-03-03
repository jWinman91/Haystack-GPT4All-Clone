import uvicorn, subprocess
from src.haystack_rag_pipeline import HaystackPipeline
from fastapi import FastAPI, HTTPException, Body, UploadFile
from typing import Annotated
from src.api_models import QueryParams


DESCRIPTION = """
"""


class App:
    def __init__(self, ip: str = "127.0.0.1", port: int = 8000) -> None:
        self._ip = ip
        self._port = port
        self._app = FastAPI(
            title="Haystack RAG Pipeline",
            description=DESCRIPTION
        )

        self._haystack = HaystackPipeline()

        self._configure_routes()

    def _configure_routes(self) -> None:
        @self._app.post("/build_index_pipeline")
        async def build_index_pipeline(embedding_model_name: str):
            self._haystack.build_index_pipeline(embedding_model_name)

        @self._app.post("/build_query_pipeline")
        async def build_query_pipeline(query_json: Annotated[QueryParams, Body(
            examples=[{
                "llm_model_name": "openai",
                "embedding_model_name": "openai",
                "prompt_template": "You are a helpful assistant. Answer questions based on the given context."
            }]
        )]
        ):
            self._haystack.build_query_pipeline(query_json.prompt_template,
                                                query_json.embedding_model_name,
                                                query_json.llm_model_name)

        @self._app.get("/get_embedding_models")
        async def get_embedding_models() -> list[str]:
            embedding_models = set(self._haystack.get_document_embedders()).union(set(self._haystack.get_text_embedders()))
            return list(embedding_models)

        @self._app.get("/get_llm_models")
        async def get_llm_models() -> list[str]:
            return self._haystack.get_llm_models()

        @self._app.post("/run_index_pipeline")
        async def run_index_pipeline(pdf_file: UploadFile):
            subprocess.call("mkdir -p tmp", shell=True)
            pdf_path = f"tmp/{pdf_file.filename}"
            with open(pdf_path, 'wb') as image:
                content = await pdf_file.read()
                image.write(content)
            self._haystack.run_index_pipeline(pdf_path)

        @self._app.post("/query_pipeline")
        async def query_pipeline(query: str) -> str:
            return self._haystack.query_pipeline(query)

    def run(self) -> None:
        """
        Run the api
        :return: None
        """
        uvicorn.run(self._app, host=self._ip, port=self._port)
        subprocess.call("rm -r tmp", shell=True)


if __name__ == "__main__":
    app = App()
    app.run()