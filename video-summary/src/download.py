import os
from pytubefix import YouTube
from moviepy.editor import AudioFileClip


def video_title(youtube_url: str) -> str:
    """
    Retrieve the title of a YouTube video.

    Examples
    --------
    >>> title = video_title("https://www.youtube.com/watch?v=SampleVideoID")
    >>> print(title)
    'Sample Video Title'
    """
    yt = YouTube(youtube_url)
    return yt.title


def download_audio(youtube_url: str, download_path: str) -> None:
    """
    Download the audio from a YouTube video.

    Examples
    --------
    >>> download_audio("https://www.youtube.com/watch?v=SampleVideoID", "path/to/save/audio.mp4")
    """
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).desc().first()
    if not audio_stream:
        raise Exception("No audio stream found")

    directory_path = os.path.dirname(download_path)
    file_name = os.path.basename(download_path)

    audio_file = audio_stream.download(output_path=directory_path, filename=file_name)

    # Проверяем размер загруженного файла
    if os.path.getsize(audio_file) < 1024:  # Проверка, что файл больше 1KB
        raise Exception("Downloaded file is too small")


def convert_mp4_to_mp3(input_path: str, output_path: str) -> None:
    """
    Convert an audio file from mp4 format to mp3.

    Examples
    --------
    >>> convert_mp4_to_mp3("path/to/audio.mp4", "path/to/audio.mp3")
    """
    audio_clip = AudioFileClip(input_path)
    audio_clip.write_audiofile(output_path, codec='libmp3lame')
    audio_clip.close()
