# app.py

import gradio as gr
import os
import tempfile
import asyncio
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from datetime import timedelta
import concurrent.futures
import zipfile
from dataclasses import dataclass
import numpy as np
import logging
from urllib.parse import urlparse

import requests

# Импортируем библиотеку автозамены
try:
    from autoreplacer import AutoReplacer, autoreplacer

    AUTOREPLACER_AVAILABLE = True
    print("AutoReplacer успешно импортирован")
except ImportError as e:
    print(f"Предупреждение: AutoReplacer не доступен: {e}")
    AUTOREPLACER_AVAILABLE = False
    autoreplacer = None

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Импортируем библиотеки для обработки аудио и AI
try:
    import gigaam

    GIGAAM_AVAILABLE = True
except ImportError:
    print("Warning: gigaam not available, will use fallback")
    GIGAAM_AVAILABLE = False

try:
    from pyannote.audio import Pipeline
    import torch

    PYANNOTE_AVAILABLE = True
except ImportError:
    print("Warning: pyannote.audio not available")
    PYANNOTE_AVAILABLE = False

# Конфигурация
MAX_DURATION_CHUNK = 29  # секунд для одного чанка транскрипции
MAX_FILE_DURATION = 9000  # 2 часа в секундах
SUPPORTED_FORMATS = ['.mp3', '.wav', '.mp4', '.avi', '.webm', '.m4a', '.flac', '.ogg']

# GPU конфигурация
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH_SIZE = 8  # Начальный размер батча для параллельной обработки
MIN_BATCH_SIZE = 1  # Минимальный размер батча при нехватке памяти


@dataclass
class TranscriptionResult:
    filename: str
    duration: float
    segments: List[Dict]
    full_text: str
    speakers_count: int


class GPUMemoryManager:
    """Менеджер GPU памяти для динамической адаптации размера батча"""

    def __init__(self):
        self.current_batch_size = MAX_BATCH_SIZE
        self.memory_errors = 0

    def get_current_batch_size(self) -> int:
        return self.current_batch_size

    def handle_memory_error(self):
        """Уменьшаем размер батча при ошибках памяти"""
        self.memory_errors += 1
        if self.current_batch_size > MIN_BATCH_SIZE:
            self.current_batch_size = max(MIN_BATCH_SIZE, self.current_batch_size // 2)
            print(f"GPU память закончилась, уменьшаем batch размер до {self.current_batch_size}")

    def clear_cache(self):
        """Очистка кэша GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_batch_size(self):
        """Сброс размера батча после успешной обработки"""
        if self.memory_errors == 0 and self.current_batch_size < MAX_BATCH_SIZE:
            self.current_batch_size = min(MAX_BATCH_SIZE, self.current_batch_size + 1)


class AudioProcessor:
    def __init__(self):
        self.gigaam_model = None
        self.diarization_pipeline = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.memory_manager = GPUMemoryManager()
        self.device = DEVICE

        print(f"Инициализация AudioProcessor на устройстве: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    async def initialize_models(self, progress_callback=None):
        """Инициализация моделей AI с улучшенной обработкой ошибок"""
        if progress_callback:
            progress_callback(0.1,
                              "Загрузка модели gigaam на GPU..." if self.device.type == 'cuda' else "Загрузка модели gigaam на CPU...")

        # Загружаем gigaam модель
        if GIGAAM_AVAILABLE:
            devices_to_try = [str(self.device)] if self.device.type == 'cuda' else ['cpu']
            if self.device.type == 'cuda':
                devices_to_try.append('cpu')  # fallback на CPU

            for device in devices_to_try:
                try:
                    self.gigaam_model = gigaam.load_model("v2_ctc", device=device)
                    print(f"Gigaam модель загружена успешно на {device}")
                    break
                except Exception as e:
                    print(f"Ошибка загрузки gigaam на {device}: {e}")
                    if device == devices_to_try[-1]:  # последняя попытка
                        self.gigaam_model = None

        if progress_callback:
            progress_callback(0.3,
                              "Загрузка модели для разделения спикеров на GPU..." if self.device.type == 'cuda' else "Загрузка модели для разделения спикеров на CPU...")

        # Загружаем pyannote для speaker diarization
        if PYANNOTE_AVAILABLE:
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.getenv("HF_TOKEN")
                )

                # Переносим модель на GPU если доступен
                if self.device.type == 'cuda':
                    self.diarization_pipeline = self.diarization_pipeline.to(self.device)

                print(f"Pyannote модель загружена успешно на {self.device}")
            except Exception as e:
                print(f"Ошибка загрузки pyannote: {e}")
                self.diarization_pipeline = None

        if progress_callback:
            progress_callback(0.5, "Модели загружены!")

        # Очищаем кэш после инициализации
        self.memory_manager.clear_cache()

    def get_audio_duration(self, file_path: str) -> float:
        """Получение длительности аудио через FFmpeg"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Файл не найден: {file_path}")

            cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", str(file_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            return duration
        except Exception as e:
            print(f"Ошибка получения длительности: {e}")
            return 0.0

    def convert_to_wav(self, input_path: str, output_path: str):
        """Конвертация медиафайла в WAV формат"""
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Входной файл не найден: {input_path}")

            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-ar", "16000",  # 16kHz sample rate
                "-ac", "1",  # mono
                "-c:a", "pcm_s16le",
                "-y", str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Проверяем, что выходной файл создан
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("Конвертация не создала выходной файл")

            return True
        except subprocess.CalledProcessError as e:
            print(f"Ошибка конвертации: {e.stderr}")
            return False
        except Exception as e:
            print(f"Ошибка конвертации: {e}")
            return False

    def extract_audio_segment(self, input_path, output_path, start_time, duration):
        """Более надежное извлечение аудиосегментов"""
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Входной файл не найден: {input_path}")

            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-ss", str(start_time),
                "-t", str(duration),
                "-ar", "16000",  # Убедитесь в одинаковой частоте дискретизации
                "-ac", "1",  # Моно
                "-c:a", "pcm_s16le",
                "-y", str(output_path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Проверяем выходной файл
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("Извлечение не создало выходной файл")

            return True
        except subprocess.CalledProcessError as e:
            print(f"Ошибка извлечения сегмента: {e.stderr}")
            return False
        except Exception as e:
            print(f"Ошибка извлечения сегмента: {e}")
            return False

    def split_audio_by_duration(self, input_path: str, output_dir: str,
                                chunk_duration: int = MAX_DURATION_CHUNK) -> List[str]:
        """Разделение аудио на чанки по времени"""
        segments = []
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Файл не найден: {input_path}")

            output_pattern = os.path.join(output_dir, "chunk_%03d.wav")
            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-f", "segment",
                "-segment_time", str(chunk_duration),
                "-c", "copy",
                "-reset_timestamps", "1",
                "-y", output_pattern
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            # Собираем созданные сегменты
            for file in sorted(os.listdir(output_dir)):
                if file.startswith("chunk_") and file.endswith(".wav"):
                    segment_path = os.path.join(output_dir, file)
                    # Проверяем, что файл не пустой
                    if os.path.exists(segment_path) and os.path.getsize(segment_path) > 0:
                        segments.append(segment_path)

        except Exception as e:
            print(f"Ошибка разделения аудио: {e}")

        return segments

    async def perform_diarization(self, audio_path: str) -> Optional[Dict]:
        """Выполнение speaker diarization с GPU оптимизацией"""
        if not self.diarization_pipeline:
            return None

        try:
            # Проверяем существование файла
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                raise FileNotFoundError(f"Аудиофайл не найден или пуст: {audio_path}")

            # Очищаем кэш перед обработкой
            self.memory_manager.clear_cache()

            loop = asyncio.get_event_loop()

            # Обработка на GPU или CPU в зависимости от устройства
            diarization = await loop.run_in_executor(
                self.executor,
                self._run_diarization_sync,
                audio_path
            )

            # Преобразуем результат в удобный формат
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                    'duration': turn.end - turn.start
                })

            return {
                'segments': segments,
                'speakers': list(set(seg['speaker'] for seg in segments))
            }

        except torch.cuda.OutOfMemoryError:
            print("GPU память закончилась при diarization")
            self.memory_manager.handle_memory_error()
            self.memory_manager.clear_cache()
            # Повторяем попытку
            return await self.perform_diarization(audio_path)
        except Exception as e:
            print(f"Ошибка diarization: {e}")
            logger.error(f"Ошибка diarization для файла {audio_path}: {e}")
            return None

    def _run_diarization_sync(self, audio_path: str):
        """Синхронная обертка для diarization"""
        try:
            return self.diarization_pipeline(audio_path)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            print(f"Ошибка в _run_diarization_sync: {e}")
            raise

    async def transcribe_audio_batch(self, audio_paths: List[str]) -> List[str]:
        """Батчевая транскрипция аудио с адаптивным размером батча"""
        if not self.gigaam_model or not audio_paths:
            return [f"[Ошибка: модель недоступна]" for _ in audio_paths]

        results = []
        current_batch_size = self.memory_manager.get_current_batch_size()

        # Обрабатываем файлы батчами
        for i in range(0, len(audio_paths), current_batch_size):
            batch = audio_paths[i:i + current_batch_size]
            batch_results = []

            try:
                # Проверяем существование всех файлов в батче
                valid_batch = []
                for path in batch:
                    if os.path.exists(path) and os.path.getsize(path) > 0:
                        valid_batch.append(path)
                    else:
                        batch_results.append("[Ошибка: файл не найден или пуст]")

                if not valid_batch:
                    results.extend(batch_results)
                    continue

                # Очищаем кэш перед каждым батчем
                self.memory_manager.clear_cache()

                # Обрабатываем батч
                loop = asyncio.get_event_loop()

                if len(valid_batch) == 1:
                    # Одиночная обработка
                    try:
                        result = await loop.run_in_executor(
                            self.executor,
                            self.gigaam_model.transcribe,
                            valid_batch[0]
                        )
                        batch_results.append(result.strip() if result else "")
                    except Exception as e:
                        logger.error(f"Ошибка транскрипции файла {valid_batch[0]}: {e}")
                        batch_results.append(f"[Ошибка транскрипции: {type(e).__name__}]")
                else:
                    # Батчевая обработка если поддерживается
                    try:
                        batch_results_inner = await loop.run_in_executor(
                            self.executor,
                            self._transcribe_batch_sync,
                            valid_batch
                        )
                        batch_results.extend([r.strip() if r else "" for r in batch_results_inner])
                    except (AttributeError, torch.cuda.OutOfMemoryError):
                        # Fallback на одиночную обработку
                        for audio_path in valid_batch:
                            try:
                                result = await loop.run_in_executor(
                                    self.executor,
                                    self.gigaam_model.transcribe,
                                    audio_path
                                )
                                batch_results.append(result.strip() if result else "")
                            except Exception as e:
                                logger.error(f"Ошибка транскрипции файла {audio_path}: {e}")
                                batch_results.append(f"[Ошибка транскрипции: {type(e).__name__}]")
                    except Exception as e:
                        # Fallback на одиночную обработку при других ошибках
                        logger.error(f"Ошибка батчевой транскрипции: {e}")
                        for audio_path in valid_batch:
                            try:
                                result = await loop.run_in_executor(
                                    self.executor,
                                    self.gigaam_model.transcribe,
                                    audio_path
                                )
                                batch_results.append(result.strip() if result else "")
                            except Exception as e2:
                                logger.error(f"Ошибка транскрипции файла {audio_path}: {e2}")
                                batch_results.append(f"[Ошибка транскрипции: {type(e2).__name__}]")

            except torch.cuda.OutOfMemoryError:
                print(f"GPU память закончилась при обработке батча размером {len(batch)}")
                self.memory_manager.handle_memory_error()
                self.memory_manager.clear_cache()

                # Обрабатываем текущий батч по одному
                for audio_path in batch:
                    try:
                        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                            result = await loop.run_in_executor(
                                self.executor,
                                self.gigaam_model.transcribe,
                                audio_path
                            )
                            batch_results.append(result.strip() if result else "")
                        else:
                            batch_results.append("[Ошибка: файл не найден или пуст]")
                    except Exception as e:
                        logger.error(f"Ошибка транскрипции {audio_path}: {e}")
                        batch_results.append(f"[Ошибка транскрипции: {type(e).__name__}]")

            except Exception as e:
                print(f"Ошибка обработки батча: {e}")
                logger.error(f"Ошибка обработки батча: {e}")
                # Fallback на одиночную обработку
                for audio_path in batch:
                    try:
                        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                            result = await loop.run_in_executor(
                                self.executor,
                                self.gigaam_model.transcribe,
                                audio_path
                            )
                            batch_results.append(result.strip() if result else "")
                        else:
                            batch_results.append("[Ошибка: файл не найден или пуст]")
                    except Exception as e2:
                        logger.error(f"Ошибка транскрипции {audio_path}: {e2}")
                        batch_results.append(f"[Ошибка транскрипции: {type(e2).__name__}]")

            results.extend(batch_results)

        return results

    def _transcribe_batch_sync(self, audio_paths: List[str]) -> List[str]:
        """Синхронная обертка для батчевой транскрипции"""
        try:
            # Проверяем, поддерживает ли модель батчевую обработку
            if hasattr(self.gigaam_model, 'transcribe_batch'):
                return self.gigaam_model.transcribe_batch(audio_paths)
            else:
                # Fallback на последовательную обработку
                results = []
                for path in audio_paths:
                    try:
                        if os.path.exists(path) and os.path.getsize(path) > 0:
                            result = self.gigaam_model.transcribe(path)
                            results.append(result)
                        else:
                            results.append("[Ошибка: файл не найден или пуст]")
                    except Exception as e:
                        logger.error(f"Ошибка транскрипции файла {path}: {e}")
                        results.append(f"[Ошибка транскрипции: {type(e).__name__}]")
                return results
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:
            print(f"Ошибка в _transcribe_batch_sync: {e}")
            raise

    async def transcribe_audio(self, audio_path: str) -> str:
        """Транскрипция одного аудио файла"""
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return "[Ошибка: файл не найден или пуст]"

        results = await self.transcribe_audio_batch([audio_path])
        return results[0] if results else f"[Ошибка транскрипции: {os.path.basename(audio_path)}]"

    def format_time(self, seconds: float) -> str:
        """Форматирование времени в MM:SS или HH:MM:SS"""
        if seconds < 3600:
            return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    async def process_single_file(self, file_path: str, progress_callback=None,
                                  original_filename=None) -> TranscriptionResult:
        """Обработка одного файла с улучшенной обработкой ошибок"""
        if original_filename:
            filename = original_filename
        else:
            filename = os.path.basename(file_path)

        if progress_callback:
            progress_callback(0.1, f"Обработка {filename}: валидация...")

        # Проверяем расширение файла
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")

        # Проверяем существование файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Конвертируем в WAV
            wav_path = os.path.join(temp_dir, "converted.wav")

            if progress_callback:
                progress_callback(0.2, f"Обработка {filename}: конвертация в WAV...")

            if not self.convert_to_wav(file_path, wav_path):
                raise ValueError("Ошибка конвертации файла")

            # Получаем длительность
            duration = self.get_audio_duration(wav_path)
            if duration > MAX_FILE_DURATION:
                raise ValueError(f"Файл слишком длинный: {duration / 3600:.1f} часов (максимум 2 часа)")

            if progress_callback:
                progress_callback(0.4,
                                  f"Обработка {filename}: разделение по спикерам на GPU..." if self.device.type == 'cuda' else f"Обработка {filename}: разделение по спикерам...")

            # Speaker diarization
            diarization_result = await self.perform_diarization(wav_path)
            speakers_count = len(diarization_result['speakers']) if diarization_result else 1

            if progress_callback:
                progress_callback(0.7,
                                  f"Обработка {filename}: транскрипция на GPU..." if self.device.type == 'cuda' else f"Обработка {filename}: транскрипция...")

            segments = []
            full_text_parts = []

            if diarization_result and len(diarization_result['segments']) > 0:
                # Подготавливаем сегменты для батчевой обработки
                segment_paths = []
                segment_info = []

                for i, seg in enumerate(diarization_result['segments']):
                    # Извлекаем сегмент аудио
                    segment_path = os.path.join(temp_dir, f"segment_{i}.wav")

                    # Используем улучшенную функцию извлечения
                    if self.extract_audio_segment(wav_path, segment_path, seg['start'], seg['duration']):
                        segment_paths.append(segment_path)
                        segment_info.append(seg)
                    else:
                        print(f"Ошибка извлечения сегмента {i}")
                        continue

                # Батчевая транскрипция всех сегментов
                if segment_paths:
                    transcriptions = await self.transcribe_audio_batch(segment_paths)

                    # Объединяем результаты
                    for seg_info, text in zip(segment_info, transcriptions):
                        if text and text.strip() and not text.startswith("[Ошибка"):
                            segments.append({
                                'start': seg_info['start'],
                                'end': seg_info['end'],
                                'speaker': seg_info['speaker'],
                                'text': text,
                                'duration': seg_info['duration']
                            })

                            speaker_num = seg_info['speaker'].replace('SPEAKER_', '') if 'SPEAKER_' in seg_info[
                                'speaker'] else seg_info['speaker']
                            time_start = self.format_time(seg_info['start'])
                            time_end = self.format_time(seg_info['end'])
                            full_text_parts.append(f"Спикер {speaker_num} [{time_start} - {time_end}]: {text}")
                else:
                    # Если не удалось извлечь сегменты, используем fallback
                    text = await self.transcribe_audio(wav_path)
                    segments.append({
                        'start': 0,
                        'end': duration,
                        'speaker': 'SPEAKER_1',
                        'text': text,
                        'duration': duration
                    })
                    time_end = self.format_time(duration)
                    full_text_parts.append(f"Спикер 1 [00:00 - {time_end}]: {text}")

            else:
                # Fallback: обрабатываем как одного спикера
                if duration <= MAX_DURATION_CHUNK:
                    # Короткий файл - транскрибируем целиком
                    text = await self.transcribe_audio(wav_path)
                    segments.append({
                        'start': 0,
                        'end': duration,
                        'speaker': 'SPEAKER_1',
                        'text': text,
                        'duration': duration
                    })
                    time_end = self.format_time(duration)
                    full_text_parts.append(f"Спикер 1 [00:00 - {time_end}]: {text}")
                else:
                    # Длинный файл - разбиваем на чанки и используем батчевую обработку
                    chunks_dir = os.path.join(temp_dir, "chunks")
                    os.makedirs(chunks_dir, exist_ok=True)

                    chunks = self.split_audio_by_duration(wav_path, chunks_dir)

                    if chunks:
                        # Батчевая транскрипция чанков
                        transcriptions = await self.transcribe_audio_batch(chunks)

                        for i, (chunk_path, text) in enumerate(zip(chunks, transcriptions)):
                            chunk_duration = self.get_audio_duration(chunk_path)
                            start_time = i * MAX_DURATION_CHUNK
                            end_time = start_time + chunk_duration

                            segments.append({
                                'start': start_time,
                                'end': end_time,
                                'speaker': 'SPEAKER_1',
                                'text': text,
                                'duration': chunk_duration
                            })

                            time_start = self.format_time(start_time)
                            time_end = self.format_time(end_time)
                            full_text_parts.append(f"Спикер 1 [{time_start} - {time_end}]: {text}")
                    else:
                        # Если не удалось создать чанки, транскрибируем весь файл
                        text = await self.transcribe_audio(wav_path)
                        segments.append({
                            'start': 0,
                            'end': duration,
                            'speaker': 'SPEAKER_1',
                            'text': text,
                            'duration': duration
                        })
                        time_end = self.format_time(duration)
                        full_text_parts.append(f"Спикер 1 [00:00 - {time_end}]: {text}")

            if progress_callback:
                progress_callback(0.9, f"Обработка {filename}: финализация...")

            # Очищаем кэш после обработки файла
            self.memory_manager.clear_cache()

            full_text = "\n".join(full_text_parts)

            # Применяем автозамену, если доступна
            if AUTOREPLACER_AVAILABLE and autoreplacer:
                try:
                    full_text = autoreplacer.apply_to_transcript(full_text)
                    logger.info("Автозамена успешно применена к транскрипции")
                except Exception as e:
                    logger.error(f"Ошибка при применении автозамены: {e}")

            return TranscriptionResult(
                filename=filename,
                duration=duration,
                segments=segments,
                full_text=full_text,
                speakers_count=speakers_count
            )


# Глобальный процессор
processor = AudioProcessor()


async def process_files(files, progress=gr.Progress()):
    """Обработка списка файлов с улучшенной обработкой ошибок"""
    if not files:
        return "Файлы не загружены", None, None

    # Инициализируем модели если нужно
    if processor.gigaam_model is None:
        progress(0.05, f"Инициализация моделей на {processor.device}...")
        await processor.initialize_models(
            lambda p, msg: progress(p * 0.1, msg)
        )

    results = []
    output_files = []

    total_files = len(files)

    for file_idx, file_info in enumerate(files):
        if hasattr(file_info, 'original_name'):
            # Это файл из ссылки с сохраненным оригинальным именем
            file_path = file_info.name
            display_name = file_info.original_name
            base_name_for_output = file_info.original_name
        else:
            # Это обычный загруженный файл
            file_path = file_info.name
            display_name = os.path.basename(file_path)
            base_name_for_output = display_name

        try:
            # Обновляем прогресс для текущего файла
            base_progress = 0.1 + (file_idx / total_files) * 0.8

            def file_progress(p, msg):
                progress(base_progress + p * (0.8 / total_files), msg)

            # Передаем оригинальное имя в process_single_file
            result = await processor.process_single_file(
                file_path,
                file_progress,
                original_filename=display_name
            )

            # Сохраняем оригинальное имя в результате
            if hasattr(result, 'filename'):
                result.filename = display_name
            results.append(result)

            # Создаем выходной файл с оригинальным именем
            name_without_ext = Path(base_name_for_output).stem
            output_filename = f"{name_without_ext}.txt"

            # Создаем путь для выходного файла
            output_path = os.path.join(tempfile.gettempdir(), output_filename)

            # Открываем файл с нужным именем
            with open(output_path, 'w', encoding='utf-8') as f:
                # Заголовок файла
                f.write(f"Транскрипция файла: {display_name}\n")
                f.write(f"Длительность: {processor.format_time(result.duration)}\n")
                f.write(f"Количество спикеров: {result.speakers_count}\n")
                f.write(f"Обработано на: {processor.device}\n")
                f.write("=" * 50 + "\n\n")

                # Основной текст
                f.write(result.full_text)

            logger.info(f"Итоговый файл сохранен: {output_path}")
            output_files.append((output_path, output_filename))

        except Exception as e:
            error_msg = f"Ошибка обработки файла {display_name}: {str(e)}"
            results.append(error_msg)
            print(error_msg)
            logger.error(error_msg)
            # Очищаем кэш после ошибки
            processor.memory_manager.clear_cache()

    progress(0.95, "Создание архива...")

    # Окончательная очистка кэша
    processor.memory_manager.clear_cache()

    # Создаем ZIP архив если файлов несколько
    if len(output_files) > 1:
        zip_path = tempfile.NamedTemporaryFile(delete=False, suffix='.zip').name
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path, arc_name in output_files:
                zipf.write(file_path, arc_name)

        device_info = f" (GPU: {torch.cuda.get_device_name()})" if torch.cuda.is_available() else " (CPU)"
        progress(1.0, f"Обработка завершена{device_info}! Обработано файлов: {len(results)}")

        # Создаем сводку
        summary = f"Обработано файлов: {len(results)} на {processor.device}{device_info}\n\n"
        for i, result in enumerate(results):
            if isinstance(result, TranscriptionResult):
                summary += f"{i + 1}. {result.filename}: ✅ ({processor.format_time(result.duration)}, {result.speakers_count} спикер(ов))\n"
            else:
                summary += f"{i + 1}. ❌ {result}\n"

        return summary, zip_path, output_files[0][0] if output_files else None
    elif len(output_files) == 1:
        device_info = f" (GPU: {torch.cuda.get_device_name()})" if torch.cuda.is_available() else " (CPU)"
        progress(1.0, f"Обработка завершена{device_info}!")
        result = results[0]
        if isinstance(result, TranscriptionResult):
            summary = f"Файл: {result.filename}\nДлительность: {processor.format_time(result.duration)}\nСпикеров: {result.speakers_count}\nУстройство: {processor.device}{device_info}\n\nСтатус: ✅ Успешно обработан"
        else:
            summary = f"❌ {result}"

        return summary, output_files[0][0], None
    else:
        return "Ошибка: не удалось обработать ни одного файла", None, None


# Создаем Gradio интерфейс
def create_interface():
    device_info = f"GPU: {torch.cuda.get_device_name()}" if torch.cuda.is_available() else "CPU"

    with gr.Blocks(title="Транскрипция аудио с разделением по спикерам (GPU)",
                   theme=gr.themes.Soft()) as demo:

        gr.Markdown(f"""
        # 🎙️ Транскрипция аудио с разделением по спикерам (GPU-оптимизированная)

        Загрузите один или несколько аудио/видео файлов для автоматической транскрипции с разделением по говорящим.

        **Устройство обработки:** {device_info}
        **Поддерживаемые форматы:** MP3, WAV, MP4, AVI, WebM, M4A, FLAC, OGG
        **Максимальная длительность:** 2 часа на файл
        **GPU ускорение:** {'✅ Включено' if torch.cuda.is_available() else '❌ Недоступно'}
        """)

        with gr.Row():
            with gr.Column(scale=2):
                files_input = gr.Files(
                    label="Загрузите аудио или видео файлы",
                    file_count="multiple",
                    file_types=SUPPORTED_FORMATS
                )

                url_input = gr.Textbox(
                    label="Или введите прямую ссылку на файл",
                    placeholder="https://example.com/audio.mp3",
                    lines=1
                )

                process_btn = gr.Button(
                    "🚀 Начать обработку на GPU" if torch.cuda.is_available() else "🚀 Начать обработку",
                    variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown(f"""
                ### ℹ️ Информация

                **Этапы обработки:**
                1. Валидация файлов
                2. Конвертация в WAV
                3. Разделение по спикерам (GPU)
                4. Батчевая транскрипция (GPU)
                5. Создание результатов

                **GPU оптимизации:**
                - Адаптивный размер батча
                - Автоматическое управление памятью
                - Параллельная обработка сегментов

                **Результат:** Текстовые файлы с таймкодами и разделением по спикерам
                """)

        with gr.Row():
            status_output = gr.Textbox(
                label="Статус обработки",
                lines=10,
                interactive=False
            )

        with gr.Row():
            with gr.Column():
                download_zip = gr.File(
                    label="📦 Скачать все результаты (ZIP)",
                    visible=False
                )

            with gr.Column():
                download_single = gr.File(
                    label="📄 Скачать результат",
                    visible=False
                )

        async def process_and_update(files, url):
            # Подготовка списка файлов для обработки
            prepared_files = list(files) if files else []

            # Загрузка файла по ссылке, если она указана
            if url and url.strip():
                try:
                    parsed = urlparse(url.strip())
                    if not parsed.scheme or not parsed.netloc:
                        raise ValueError("Некорректная ссылка")

                    url_path = parsed.path
                    # Извлекаем базовое имя файла из пути url
                    base_filename = os.path.basename(url_path)
                    if not base_filename:
                        raise ValueError("Не удалось извлечь имя файла из ссылки")

                    name_part, ext_part = os.path.splitext(base_filename)
                    ext = ext_part.lower()

                    # Проверяем расширение
                    if ext and ext not in SUPPORTED_FORMATS:
                        raise ValueError(f"Неподдерживаемый формат файла по ссылке: {ext}")
                    if not ext:
                        ext = ".mp3"  # расширение по умолчанию
                        # Если расширения не было, добавляем его к базовому имени для согласованности
                        base_filename = name_part + ext

                    # Скачиваем файл
                    resp = requests.get(url.strip(), stream=True, timeout=60)
                    resp.raise_for_status()

                    temp_dir = tempfile.mkdtemp()
                    temp_file_path = os.path.join(temp_dir, base_filename)

                    with open(temp_file_path, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    class _URLFile:
                        def __init__(self, name, original_name):
                            self.name = name
                            self.original_name = original_name

                    # Передаем путь к файлу и его оригинальное имя
                    prepared_files.append(_URLFile(temp_file_path, base_filename))

                except Exception as e:
                    error_msg = f"❌ Ошибка загрузки файла по ссылке: {str(e)}"
                    logger.error(error_msg)
                    return error_msg, None, None, gr.update(visible=False), gr.update(visible=False)

            if not prepared_files:
                return "❌ Пожалуйста, загрузите файлы или укажите ссылку", None, None, gr.update(
                    visible=False), gr.update(visible=False)

            try:
                summary, zip_file, single_file = await process_files(prepared_files)

                zip_visible = zip_file is not None
                single_visible = single_file is not None and zip_file is None

                if zip_file:
                    logger.info(f"ZIP архив сохранен: {zip_file}")
                elif single_file:
                    logger.info(f"Итоговый файл сохранен: {single_file}")

                return (
                    summary,
                    zip_file,
                    single_file,
                    gr.update(visible=zip_visible),
                    gr.update(visible=single_visible)
                )
            except Exception as e:
                error_msg = f"❌ Критическая ошибка: {str(e)}"
                logger.error(f"Критическая ошибка: {e}")
                return error_msg, None, None, gr.update(visible=False), gr.update(visible=False)

        process_btn.click(
            fn=process_and_update,
            inputs=[files_input, url_input],
            outputs=[status_output, download_zip, download_single, download_zip, download_single]
        )

    return demo


if __name__ == "__main__":
    # Инициализируем приложение
    demo = create_interface()
    demo.launch(share=True)