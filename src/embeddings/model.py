from fastembed.text.onnx_embedding import OnnxTextEmbedding
from pathlib import Path
from src.embeddings import wrappers  # noqa


class LocalEmbeddingsModel(OnnxTextEmbedding):
    def __init__(
        self,
        model_id: str = "BAAI/bge-base-en",
        model_name: str = "bge-base-en",
        dim: int = 768,
        cache_dir: str = "/tmp/.lightdb/.cache/.models/model--BAAI--bge-base-en",
        threads: int = 6,
    ):
        self.cache_dir = cache_dir
        self._model_dir = Path(
            self.download_files_from_huggingface(
                hf_source_repo=model_id, cache_dir=cache_dir, local_files_only=True
            )
        )
        self.model_name = model_name
        self.providers = None
        self.cuda = False
        self.device_ids = None
        self.device_id = None
        self.model_description = {
            "model": model_id,
            "dim": dim,
            "description": "Text embeddings model",
            "license": "mit",
            "size_in_GB": 0.42,
            "sources": [{"url": "https://storage.googleapis.com"}],
            "model_file": "model_optimized.onnx",
        }
        self.threads = threads
