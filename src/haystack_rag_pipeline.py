import os

from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.writers import DocumentWriter
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import *
from haystack.components.generators.chat import *

from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.embedders.fastembed import FastembedTextEmbedder, FastembedDocumentEmbedder


class HaystackPipeline:
    PROMPT_TEMPLATE_ADDITION = """
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:
    """
    def __init__(self, split_length: int = 10, split_overlap: int = 2, split_by: str = "sentence"):
        self._split_length = split_length
        self._split_overlap = split_overlap
        self._split_by = split_by

        self._document_store = InMemoryDocumentStore()

        self._document_embedder = {
            "fast_embed_small": {
                "object": FastembedDocumentEmbedder,
                "params": {
                    "model": "BAAI/bge-small-en-v1.5",
                    "parallel": 0,
                }
            },
            "sentence_transformer-mini": {
                "object": SentenceTransformersDocumentEmbedder,
                "params": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            },
            "sentence_transformer-nomic": {
                "object": SentenceTransformersDocumentEmbedder,
                "params": {
                    "model": "nomic-ai/nomic-embed-text-v1",
                    "model_kwargs": {"trust_remote_code": True},
                    "tokenizer_kwargs": {"trust_remote_code": True},
                    "config_kwargs": {"trust_remote_code": True}
                }
            },
            "openai": {
                "object": OpenAIDocumentEmbedder,
                "params": {}
            }
        }

        self._text_embedder = {
            "sentence_transformer-mini": {
                "object": SentenceTransformersTextEmbedder,
                "params": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            },
            "sentence_transformer-nomic": {
                "object": SentenceTransformersTextEmbedder,
                "params": {
                    "model": "nomic-ai/nomic-embed-text-v1",
                    "model_kwargs": {"trust_remote_code": True},
                    "tokenizer_kwargs": {"trust_remote_code": True},
                    "config_kwargs": {"trust_remote_code": True}
                }
            },
            "fast_embed_small": {
                "object": FastembedTextEmbedder,
                "params": {
                    "model": "BAAI/bge-small-en-v1.5"
                }
            },
            "openai": {
                "object": OpenAITextEmbedder,
                "params": {}
            }
        }

        self._llm_model = {
            "huggingface-qwen-2.5": {
                "object": HuggingFaceLocalChatGenerator,
                "params": {
                    "model": "Qwen/Qwen2.5-1.5B-Instruct"
                }
            },
            "ollama-gemma2": {
                "object": OllamaChatGenerator,
                "params": {
                    "model": "gemma2",
                    "url": "http://localhost:11434"
                }
            },
            "ollama-llama3": {
                "object": OllamaChatGenerator,
                "params": {
                    "model": "llama3:8b",
                    "url": "http://localhost:11434"
                }
            },
            "openai": {
                "object": OpenAIChatGenerator,
                "params": {}
            }

        }

        self._preprocessing_pipeline = None
        self._retriever_pipeline = None

    def get_document_embedders(self):
        return list(self._document_embedder.keys())

    def get_text_embedders(self):
        return list(self._text_embedder.keys())

    def get_llm_models(self):
        return list(self._llm_model.keys())

    def build_index_pipeline(self, document_embedder_str: str):
        self._preprocessing_pipeline = Pipeline()

        document_embedder_config = self._document_embedder.get(document_embedder_str, "openai")
        document_embedder = document_embedder_config["object"](**document_embedder_config["params"])
        if document_embedder_str != "openai":
            document_embedder.warm_up()

        self._preprocessing_pipeline.add_component(instance=FileTypeRouter(mime_types=["application/pdf"]),
                                                   name="file_type_router")
        self._preprocessing_pipeline.add_component(instance=PyPDFToDocument(), name="pypdf_converter")
        self._preprocessing_pipeline.add_component(instance=DocumentJoiner(), name="document_joiner")
        self._preprocessing_pipeline.add_component(instance=DocumentCleaner(), name="document_cleaner")
        self._preprocessing_pipeline.add_component(instance=DocumentSplitter(split_by=self._split_by,
                                                                             split_length=self._split_length,
                                                                             split_overlap=self._split_overlap),
                                                   name="document_splitter")
        self._preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
        self._preprocessing_pipeline.add_component(instance=DocumentWriter(self._document_store),
                                                   name="document_writer")

        self._preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
        self._preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
        self._preprocessing_pipeline.connect("document_joiner", "document_cleaner")
        self._preprocessing_pipeline.connect("document_cleaner", "document_splitter")
        self._preprocessing_pipeline.connect("document_splitter", "document_embedder")
        self._preprocessing_pipeline.connect("document_embedder", "document_writer")

    def build_query_pipeline(self, prompt_template: str, text_embedder_str: str, llm_model_str: str):
        text_embedder_config = self._text_embedder.get(text_embedder_str, "openai")
        text_embedder = text_embedder_config["object"](**text_embedder_config["params"])
        if text_embedder_str != "openai":
            text_embedder.warm_up()

        llm_model_config = self._llm_model.get(llm_model_str, "openai")
        llm_model = llm_model_config["object"](**llm_model_config["params"])

        self._retriever_pipeline = Pipeline()
        retriever = InMemoryEmbeddingRetriever(self._document_store)
        prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(prompt_template +
                                                                           self.PROMPT_TEMPLATE_ADDITION)])

        self._retriever_pipeline.add_component(instance=text_embedder, name="text_embedder")
        self._retriever_pipeline.add_component(instance=retriever, name="retriever")
        self._retriever_pipeline.add_component(instance=prompt_builder, name="prompt_builder")
        self._retriever_pipeline.add_component(instance=llm_model, name="llm_model")

        self._retriever_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        self._retriever_pipeline.connect("retriever", "prompt_builder.documents")
        self._retriever_pipeline.connect("prompt_builder.prompt", "llm_model.messages")

    def run_index_pipeline(self, document_path: str):
        self._preprocessing_pipeline.run({"file_type_router": {"sources": [document_path]}})

    def query_pipeline(self, query: str) -> str:
        response = self._retriever_pipeline.run({"text_embedder": {"text": query},
                                                  "prompt_builder": {"question": query}},
                                                 include_outputs_from = {"prompt_builder"})
        print(response)
        return response["llm_model"]["replies"][0]._content[0].text
