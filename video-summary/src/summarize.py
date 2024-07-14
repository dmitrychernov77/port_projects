from openai import OpenAI


def summary_prompt(input_text: str) -> str:
    """
    Build prompt using input text of the video.
    """
    prompt = f"""
    Твоя задача сгенерировать короткое саммари для расшифровки видео с YouTube.

    Сделай суммаризацию для текста ниже, заключенного в тройные квадратные скобки, минимум 30 слов.
    Сфокусируйся на главных аспектах о чем говорится в видео.

    Текст для суммаризации ```{input_text}```
    """
    return prompt


def summarize_text(input_text: str) -> str:
    """
    Summarize input text of the video.

    Examples
    --------
    >>> summary = summarize_text(video_text)
    >>> print(summary)
    'This video explains...'
    """
    # Send request to OpenAI
    promt = summary_prompt(input_text)

    client = OpenAI(
        api_key="sk-proj-p9LhmH9p2IOLX8FNE2SvT3BlbkFJaoF3FvhLMa2ztcBiD9I7"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": promt}],
    )
    summary = response.choices[0].message.content
    return summary
