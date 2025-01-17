import wrapt
from fastembed.text.onnx_embedding import OnnxTextEmbedding
from fastembed import TextEmbedding


def wrapped_download(*args, **kwargs):
    model_path = kwargs["cache_dir"]
    return model_path


@wrapt.decorator
def load_model_wrapper(wrapped, instance, args, kwargs):  # noqa
    # Call the replacement function instead of the original
    return wrapped_download(*args, **kwargs)


OnnxTextEmbedding.download_files_from_huggingface = load_model_wrapper(
    TextEmbedding.download_files_from_huggingface
)
