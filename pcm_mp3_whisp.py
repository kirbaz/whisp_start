import torch
import whisper
from pydub import AudioSegment
import soundfile as sf


def pcm_to_wav(pcm_file_path: str, wav_file_path: str):
    # Загружаем аудиофайл PCM
    pcm_audio, sample_rate = sf.read(pcm_file_path)

    # Сохраняем аудиофайл в формате WAV
    sf.write(wav_file_path, pcm_audio, sample_rate)


def transcribe_mp3(file_path: str, model_name="base") -> str:
    # Проверяем доступность GPU и CUDA-устройства
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        device = torch.device("cuda")
        model_path = f"saved_models/{model_name}.pt"
    else:
        device = torch.device("cpu")
        model_path = f"saved_models/{model_name}_cpu.pt"

    try:
        # Загружаем модель с сохраненного файла на устройство (GPU или CPU)
        model = torch.load(model_path, map_location=device)
    except FileNotFoundError:
        # Если сохраненная модель не найдена, загружаем модель с помощью whisper.load_model
        model = whisper.load_model(model_name).to(device)
        # Сохраняем модель на диск для последующих запусков программы
        torch.save(model, model_path)
    # Загружаем аудиофайл MP3
    audio = AudioSegment.from_mp3(file_path)

    # Преобразуем аудиофайл в формат PCM
    audio = audio.set_channels(1)  # Преобразуем в моно
    audio.export("save_audio/temp.wav", format="wav")  # Экспортируем во временный WAV-файл

    # Загружаем аудиофайл PCM
    pcm_audio = whisper.load_audio("save_audio/temp.wav")

    # Преобразуем numpy.ndarray в torch.Tensor
    audio = torch.from_numpy(pcm_audio)

    # Перемещаем аудиофайл на устройство (GPU или CPU)
    audio = audio.to(device)

    # Транскрибируем аудиофайл
    result = model.transcribe(audio)

    # Если используется GPU, перемещаем результат обратно на CPU
    if device.type == "cuda":
        result["text"] = result["text"].to("cpu")

    return result["text"]

text = transcribe_mp3(r'c:\py_projects\first\venv\save_audio\questions.mp3')
print(text)
