import whisper


def transcribe(file_path: str, model_name="base") -> str:
    """
    Transcribe input audio file.

    Examples
    --------
    >>> text = transcribe(".../audio.mp3")
    >>> print(text)
    'This text explains...'
    """
    model = whisper.load_model(model_name)
    result = model.transcribe(file_path, fp16=False)
    return result["text"]
