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

`HF_TOKEN` нужен только для первой загрузки pyannote. Перед запуском команды скачивания примите условия `pyannote/speaker-diarization-3.1` и вложенной gated-модели `pyannote/segmentation-3.0` на Hugging Face и внесите токен в `.env`. Если полный локальный кэш уже создан, повторный запуск без `--force` не требует токен.

## Где хранятся модели

В Docker Compose подключен постоянный host volume:

```yaml
./models:/opt/gigascribe/models
```

Явные локальные пути:

- pyannote snapshot: `/opt/gigascribe/models/pyannote-speaker-diarization-3.1`;
- Hugging Face cache для вложенных моделей pyannote: `/opt/gigascribe/models/huggingface`;
- GigaAM cache: `/opt/gigascribe/models/gigaam-cache`;
- checkpoint GigaAM `v2_ctc`: `/opt/gigascribe/models/gigaam-cache/v2_ctc.ckpt`;
- marker GigaAM: `/opt/gigascribe/models/gigaam/v2_ctc/.gigaam-ready.json`.

Модели не входят в Docker-образ. `docker compose down/up`, пересборка образа и удаление контейнера не удаляют `./models`. Для полного удаления моделей удалите каталог вручную:

```bash
rm -rf ./models
```

## Проверка загрузки

```bash
docker compose run --rm gigascribe test -f /opt/gigascribe/models/pyannote-speaker-diarization-3.1/.pyannote-ready.json
docker compose run --rm gigascribe test -s /opt/gigascribe/models/gigaam-cache/v2_ctc.ckpt
docker compose run --rm gigascribe test -f /opt/gigascribe/models/gigaam/v2_ctc/.gigaam-ready.json
```

## Обычный запуск без интернета

Runtime mode использует `HF_HUB_OFFLINE=1`, `HF_HOME=/opt/gigascribe/models/huggingface`, локальные marker-файлы и предварительную проверку checkpoint GigaAM до вызова `gigaam.load_model(..., download_root="/opt/gigascribe/models/gigaam-cache")`. После первой загрузки можно запускать сервис без доступа в интернет:

```bash
docker compose up -d
```

Если GigaAM отсутствует, задание завершается `status="error"`, потому что транскрибация невозможна. Если pyannote отсутствует, сервис выводит предупреждение и работает в документированном fallback-режиме с одним спикером.

## Download mode

Скрипт `scripts/download_models.py` временно переводит Hugging Face Hub в online mode для загрузки, явно передает GigaAM `download_root=/opt/gigascribe/models/gigaam-cache`, создает pyannote pipeline online, затем включает offline mode и повторно создает pipeline без сети. Исправные модели не перекачиваются без `--force`. Marker создается только после проверки реального checkpoint/offline pipeline.

Принудительно обновить обе модели безопасно: GigaAM скачивается во временный каталог, проверяется и только затем атомарно заменяет рабочий checkpoint; старая модель остается на месте при ошибке.

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

Проект использует Python 3.11, поэтому backport-пакеты `asyncio`, `dataclasses`, `pathlib2`, `zipfile36` удалены. Критические ML-зависимости ограничены совместимой группой: `torch`/`torchaudio` одной версии, `pyannote.audio` 3.1.x для модели diarization 3.1, `numpy<2` для совместимости с аудио/ML стеком, `huggingface_hub<0.30` из-за используемого API `snapshot_download(local_dir=...)`. GigaAM вызывается только через официальный `gigaam.load_model(model_name, device=..., download_root=...)`; локальный path API в коде не используется, потому что он не подтвержден как стабильный.


## Права bind mounts

Контейнер не запускает сервер от root. UID/GID пользователя `gigascribe` задаются build args `HOST_UID` и `HOST_GID` (см. `.env.example`). Для Linux-хоста обычно достаточно:

```bash
echo "HOST_UID=$(id -u)" >> .env
echo "HOST_GID=$(id -g)" >> .env
docker compose build
```

Так пользователь внутри контейнера совпадает с владельцем `./data` и `./models` на хосте и может создать `data/users.json`, сохранить загруженный файл, результат транскрибации и модели без `chmod 777`.

## Healthcheck

- `GET /health/live` проверяет только живой HTTP-процесс.
- `GET /health/ready` проверяет запись в `data` и `models`, наличие `ffmpeg/ffprobe` и локального checkpoint GigaAM. Pyannote отражается в JSON, но его отсутствие не делает сервис полностью недоступным, потому что поддерживается fallback одного спикера.

## Полное удаление моделей

```bash
docker compose down
rm -rf ./models/*
```

После этого снова выполните команду первой установки. `docker compose down` и пересборка образа сами по себе содержимое `./models` не удаляют.
