# GigaScribe

GigaScribe — локальный сервис транскрибации: FFmpeg нормализует аудио, pyannote при наличии делает diarization, GigaAM выполняет распознавание, затем применяется autoreplacer.

## Первая установка

```bash
cp .env.example .env
mkdir -p data models
docker compose build
docker compose run --rm gigascribe python scripts/download_models.py --models-dir /opt/gigascribe/models
docker compose up -d
```

`HF_TOKEN` нужен только для первой загрузки pyannote. Перед запуском команды скачивания примите условия `pyannote/speaker-diarization-3.1` на Hugging Face и внесите токен в `.env`.

## Где хранятся модели

В Docker Compose подключен постоянный host volume:

```yaml
./models:/opt/gigascribe/models
```

Явные локальные пути:

- pyannote: `/opt/gigascribe/models/pyannote-speaker-diarization-3.1`;
- GigaAM: штатный кэш загрузчика внутри `/opt/gigascribe/models` (`HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`) плюс marker `/opt/gigascribe/models/gigaam/<model>/.gigaam-ready.json`.

Модели не входят в Docker-образ. `docker compose down/up`, пересборка образа и удаление контейнера не удаляют `./models`. Для полного удаления моделей удалите каталог вручную:

```bash
rm -rf ./models
```

## Проверка загрузки

```bash
docker compose run --rm gigascribe test -f /opt/gigascribe/models/pyannote-speaker-diarization-3.1/.pyannote-ready.json
docker compose run --rm gigascribe test -f /opt/gigascribe/models/gigaam/v2_ctc/.gigaam-ready.json
```

## Обычный запуск без интернета

Runtime mode использует `HF_HUB_OFFLINE=1`, `HF_HOME=/opt/gigascribe/models/huggingface`, `TORCH_HOME=/opt/gigascribe/models/torch` и локальные marker-файлы. После первой загрузки можно запускать сервис без доступа в интернет:

```bash
docker compose up -d
```

Если GigaAM отсутствует, задание завершается `status="error"`, потому что транскрибация невозможна. Если pyannote отсутствует, сервис выводит предупреждение и работает в документированном fallback-режиме с одним спикером.

## Download mode

Скрипт `scripts/download_models.py` временно переводит Hugging Face Hub в online mode для загрузки и не перекачивает исправные модели без `--force`.

Принудительно обновить обе модели:

```bash
docker compose run --rm gigascribe python scripts/download_models.py --models-dir /opt/gigascribe/models --force
```

Скачать только pyannote:

```bash
docker compose run --rm gigascribe python scripts/download_models.py --models-dir /opt/gigascribe/models --skip-gigaam
```

Скачать/прогреть только GigaAM:

```bash
docker compose run --rm gigascribe python scripts/download_models.py --models-dir /opt/gigascribe/models --skip-pyannote
```

## Пользователи

Администратор по умолчанию: `admin`, пароль задается `GIGASCRIBE_ADMIN_PASSWORD` в `.env`. Создавать локальных пользователей может только администратор:

```bash
curl -u admin:change-me -F username=ivan -F password='strong-password' http://localhost:8000/admin/users
```

## GPU NVIDIA

CPU поддерживается по умолчанию. Для GPU установите NVIDIA Container Toolkit и раскомментируйте пример `gpus: all` в `docker-compose.yml`. Torch/GigaAM автоматически используют CUDA, если она доступна, иначе выполняется CPU fallback.

## Зависимости

Проект использует Python 3.11, поэтому backport-пакеты `asyncio`, `dataclasses`, `pathlib2`, `zipfile36` удалены. Критические ML-зависимости ограничены совместимой группой: `torch`/`torchaudio` одной версии, `pyannote.audio` 3.1.x для модели diarization 3.1, `numpy<2` для совместимости с аудио/ML стеком, `huggingface_hub<0.30` из-за используемого API `snapshot_download(local_dir=...)`. GigaAM вызывается только через официальный `gigaam.load_model(model_name, device=...)`; локальный path API в коде не используется, потому что он не подтвержден как стабильный.
