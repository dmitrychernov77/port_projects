import os
import re
from uuid import uuid4

import streamlit as st

from src.download import convert_mp4_to_mp3, download_audio, video_title
from src.transcribe import transcribe
from src.summarize import summarize_text


def main():
    st.title("Видео Суммаризатор")

    # Paste url to youtube video
    youtube_url = st.text_input("Вставьте ссылку на видеоролик в youtube:")

    # Regex check youtube url
    if re.match(r"^https://www.youtube.com/watch\?v=[a-zA-Z0-9_-]*$", youtube_url):
        # Display video
        st.video(youtube_url)

        transcribe_button = st.empty()
        title_placeholder = st.empty()
        progress_placeholder = st.empty()

        # Button to download audio from youtube video
        if transcribe_button.button("Суммаризировать видео"):
            # Download audio
            try:
                transcribe_button.empty()

                title_placeholder.title(video_title(youtube_url))

                progress_placeholder.text("Скачиваю видео...")

                # All mp4 and mp3 files will be saved in the runtimes folder
                # each mp4 and mp3 will have a unique runtime_id name
                # example: runtimes/cbec467e-71d9-4a3f-a3d3-406fa3438728.mp3
                runtime_id = str(uuid4())
                download_path = "runtimes"
                os.makedirs(download_path, exist_ok=True)
                mp4_file_path = os.path.join(download_path, f"{runtime_id}.mp4")
                mp3_file_path = os.path.join(download_path, f"{runtime_id}.mp3")

                # Download audio to runtimes/ folder
                download_audio(youtube_url, mp4_file_path)

                # Convert mp4 to mp3
                convert_mp4_to_mp3(mp4_file_path, mp3_file_path)

            except Exception as e:
                print(e)
                st.error("Пожалуйста, предоставьте корректную ссылку на видео!")
                transcribe_button.empty()
                title_placeholder.empty()
                progress_placeholder.empty()
                st.stop()

            # Transcribe
            try:
                progress_placeholder.text("Распознавание аудио...")

                video_text = transcribe(mp3_file_path, model_name="base")
            except Exception as e:
                print(e)
                st.error("Ошибка распознавания. Пожалуйста, попробуйте еще раз!")
                title_placeholder.empty()
                progress_placeholder.empty()
                st.stop()

            # Summarize
            try:
                assert os.environ["OPENAI_API_KEY"], "OPENAI_API_KEY not found!"

                progress_placeholder.text("Суммаризация...")

                # Summarize text
                summary = summarize_text(video_text)

                st.text_area("Результат", summary, height=300)
            except Exception as e:
                print(e)
                st.error("Ошибка суммаризации. Пожалуйста, попробуйте еще раз!")
                title_placeholder.empty()
                progress_placeholder.empty()
                st.stop()

            progress_placeholder.empty()


if __name__ == "__main__":
    main()
