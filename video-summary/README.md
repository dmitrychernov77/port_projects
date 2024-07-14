# Сервис суммаризации видео

    - по ссылке из YouTube достает из видео аудиодорожку
    - преобразует аудио в текст с помощью Whisper
    - суммаризирует текст в короткое описание с помощью ChatGPT-3.5 через OpenAI API

## Установка

[Установите ffmpeg](https://www.ffmpeg.org/download.html)

```bash
pip install -r requirements.txt
```

## Как запустить сервис

```bash
export OPENAI_API_KEY="your key"

streamlit run app.py
```

URL: http://0.0.0.0:3333 или http://localhost:3333
