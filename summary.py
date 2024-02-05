from pytube import YouTube
from moviepy.editor import AudioFileClip
import os
import whisper
import torch


def video_title(youtube_url: str) -> str:
    """
    Retrieve the title of a YouTube video.

    Examples
    --------
    >>> title = video_title("https://www.youtube.com/watch?v=SampleVideoID")
    >>> print(title)
    'Sample Video Title'
    https://www.youtube.com/watch?v=iqbsHiSnZQE
    """
    yt = YouTube(youtube_url)
    video_title = yt.title
    return video_title

# video_title("https://www.youtube.com/watch?v=iqbsHiSnZQE")

def download_audio(youtube_url: str, output_path: str) -> None:
    """
    Download the audio from a YouTube video.

    Examples
    --------
    >>> download_audio("https://www.youtube.com/watch?v=SampleVideoID", "path/to/save/audio.mp4")
    """
    yt = YouTube(youtube_url)
    audio = yt.streams.filter(only_audio=True).first()
    output_filename = os.path.basename(output_path)
    audio.download(output_path=os.path.dirname(output_path), filename=output_filename)

# download_audio("https://www.youtube.com/watch?v=IQUt9qTsQ0s", "save_audio")

def convert_mp4_to_mp3(input_path: str, output_path: str) -> None:
    """
    Convert an audio file from mp4 format to mp3.

    Examples
    --------
    >>> convert_mp4_to_mp3("path/to/audio.mp4", "path/to/audio.mp3")
    """
    # YOUR CODE HERE
    audio = AudioFileClip(input_path)
    audio.write_audiofile(output_path, codec="mp3")

# convert_mp4_to_mp3(r"c:\py_projects\first\venv\save_audio\questions.mp4",
#                    r"c:\py_projects\first\venv\save_audio\questions.mp3")

def transcribe(file_path: str, model_name="base") -> str:
    """
    Transcribe input audio file.

    Examples
    --------
    >>> text = transcribe(".../audio.mp3")
    >>> print(text)
    'This text explains...'
    """
    # Проверяем доступность GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель с помощью whisper.load_model
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        model_path = f"saved_models/{model_name}.pt"
    else:
        model_path = f"saved_models/{model_name}_cpu.pt"
    # model = whisper.load_model(model_name, device=device)
    try:
        # Загружаем модель с сохраненного файла на устройство (GPU или CPU)
        model = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        # Если сохраненная модель не найдена, загружаем модель с помощью whisper.load_model
        model = whisper.load_model(model_name).to(device)
        # Сохраняем модель на диск для последующих запусков программы
        torch.save(model, model_path)
    # Загружаем аудиофайл с помощью whisper.load_audio
    audio = whisper.load_audio(file_path)

    # Преобразуем numpy.ndarray в torch.Tensor
    audio = torch.from_numpy(audio)

    # Перемещаем аудиофайл на устройство (GPU или CPU)
    audio = audio.to(device)

    # Транскрибируем аудиофайл
    result = model.transcribe(audio)

    # Если используется GPU, перемещаем результат обратно на CPU
    if device.type == "cuda":
        result["text"] = result["text"].to("cpu")

    return result["text"]


# def transcribe(file_path: str, model_name="base") -> str:
#     """
#     Transcribe input audio file.
#
#     Examples
#     --------
#     >>> text = transcribe(".../audio.mp3")
#     >>> print(text)
#     'This text explains...'
#     """
#     # YOUR CODE HERE
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Проверяем наличие сохраненной модели
#     if torch.cuda.is_available() and torch.cuda.device_count() > 0:
#         model_path = f"saved_models/{model_name}.pt"
#     else:
#         model_path = f"saved_models/{model_name}_cpu.pt"
#
#     try:
#         # Загружаем модель с сохраненного файла на устройство (GPU или CPU)
#         model = torch.load(model_path, map_location=device)
#     except FileNotFoundError:
#         # Если сохраненная модель не найдена, загружаем модель с помощью whisper.load_model
#         model = whisper.load_model(model_name).to(device)
#         # Сохраняем модель на диск для последующих запусков программы
#         torch.save(model, model_path)
#
#     # Загружаем аудиофайл на устройство (GPU или CPU)
#     audio = whisper.load_audio(file_path)
#
#     # Преобразуем numpy.ndarray в torch.Tensor
#     audio = torch.from_numpy(audio)
#
#     # Перемещаем аудиофайл на устройство (GPU или CPU)
#     audio = audio.to(device)
#
#     # Транскрибируем аудиофайл
#     result = model.transcribe(audio, device=device)
#
#     # Если используется GPU, перемещаем результат обратно на CPU
#     if device.type == "cuda":
#         result["text"] = result["text"].to("cpu")
#
#     return result["text"]


# text = transcribe(r"save_audio\questions.mp3")
# print(text)
