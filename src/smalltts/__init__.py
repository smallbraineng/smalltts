def __getattr__(name):
    if name == "SmallTTS":
        from .infer.onnx import SmallTTS
        return SmallTTS
    raise AttributeError(name)


def hello() -> str:
    return "Hello from smalltts!"
