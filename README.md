# GigaScribe

Локальный сервис транскрибации для Ubuntu/Docker Compose с режимами CPU и NVIDIA GPU.

## Поддерживаемые модели

ASR: `GigaAM v2_ctc`, `GigaAM v3_ctc`. Диаризация: `pyannote speaker-diarization-3.1`, `pyannote speaker-diarization-community-1`, `Без диаризации`. Выбор сохраняется в `models/settings.json`; новые задания получают снимок настроек, уже запущенные задания не переключаются молча.

## Быстрый старт CPU

```bash
cp .env.example .env  # если шаблон отсутствует, создайте .env вручную
printf 'GIGASCRIBE_SECRET_KEY=%s\nGIGASCRIBE_ADMIN_PASSWORD=%s\nHOST_UID=%s\nHOST_GID=%s\n' "$(openssl rand -hex 32)" "change-this-password" "$(id -u)" "$(id -g)" > .env
scripts/preflight.sh cpu
docker compose up -d --build
```

CPU-образ использует `requirements-cpu.txt` и не скачивает CUDA wheels.

## Устойчивость сборки к обрывам сети

Dockerfile использует BuildKit cache mount с постоянным идентификатором `gigascribe-pip-cache` для `/root/.cache/pip`. Этот каталог управляется builder'ом Docker, а не находится внутри финального образа: он сохраняет HTTP-кэш и полностью скачанные wheels между запусками `docker compose build` на том же builder'е. Поэтому уже завершённые загрузки, включая `torch`, `torchaudio`, `nvidia-cudnn` и `nvidia-cublas`, при неизменных версиях берутся из кэша, а не загружаются повторно.

Каждая установка зависимостей использует таймаут 600 секунд, 20 попыток сетевого подключения и до 50 попыток докачки в рамках запуска `pip`. Pip обновляется перед установкой зависимостей, чтобы поддерживалась опция `--resume-retries`. Если соединение оборвалось во время работающего `pip`, он запрашивает оставшийся диапазон файла; после успешного завершения wheel сохраняется в BuildKit cache. Если сам процесс сборки принудительно остановлен до завершения конкретного wheel, pip удаляет свой временный неполный файл, поэтому следующий запуск не может докачать именно этот незавершённый файл: он скачает только этот wheel заново, сохранив все уже завершённые wheels. Это ограничение pip, а не очистка BuildKit cache.

Для работы cache mount нужен BuildKit. В актуальном Docker Compose v2 он обычно включён по умолчанию; при старой конфигурации запускайте сборку так:

```bash
DOCKER_BUILDKIT=1 docker compose build
DOCKER_BUILDKIT=1 docker compose up -d --build
```

Не используйте `docker compose build --no-cache`: эта опция заставляет Docker повторно выполнить слои сборки. Также не очищайте кэш builder'а (`docker builder prune`), если хотите сохранить уже скачанные wheels.

Порядок слоёв намеренно разделён: PyTorch, зависимости моделей, базовые зависимости, `gigaam --no-deps`, затем исходный код. Поэтому изменение файлов приложения не инвалидирует слои установки зависимостей.

## Быстрый старт NVIDIA GPU / RTX 50xx

1. Проверьте GPU: `nvidia-smi`.
2. Установите драйвер NVIDIA и NVIDIA Container Toolkit.
3. Настройте runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

4. Проверьте Docker GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

5. Запустите preflight и сервис:

```bash
scripts/preflight.sh gpu
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
```

GPU-сборка использует `requirements-cu128.txt` с `torch==2.7.0`, `torchaudio==2.7.0` и индексом `https://download.pytorch.org/whl/cu128`. Для RTX 5060 Ti проверяйте, что `torch.cuda.get_arch_list()` содержит `sm_120`, а реальная CUDA-операция проходит. Версия CUDA в `nvidia-smi` и `torch.version.cuda` может отличаться; это нормально при достаточно новом драйвере.

## Проверка PyTorch внутри контейнера

```bash
docker compose exec gigascribe python - <<'PY'
import torch, torchaudio
print(torch.__version__, torchaudio.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_arch_list())
x = torch.randn((2048, 2048), device='cuda')
y = x @ x
torch.cuda.synchronize()
print('cuda real test ok')
PY
```

Если RTX 5060 Ti обнаружена, но `sm_120` отсутствует, установлена несовместимая сборка PyTorch; используйте CUDA 12.8 wheel.

## Скачивание моделей

```bash
python scripts/download_models.py --models-dir ./models --gigaam-model v2_ctc --skip-pyannote
python scripts/download_models.py --models-dir ./models --gigaam-model v3_ctc --skip-pyannote
HF_TOKEN=... python scripts/download_models.py --models-dir ./models --skip-gigaam
python scripts/download_models.py --models-dir ./models --offline
```

`gigaam==0.1.0` устанавливается в Docker через `pip install --no-deps`, потому что metadata пакета ограничивает `torch<=2.5.1`. Это временный обход ограничения metadata, а не скрытие конфликта; совместимость должна подтверждаться smoke-тестами импорта, загрузки `v2_ctc`/`v3_ctc`, CPU/CUDA-инференса, длинного аудио и повторной загрузки.

Для локального pyannote используется путь к `config.yaml`, marker `.pyannote-ready.json` создаётся только после загрузочной проверки. HF token лучше передавать одноразово или через secret; не публикуйте вывод `docker compose config`, он может раскрыть `.env`.

## API и UI

Доступны `/api/models`, `/api/models/status`, `/api/models/download`, `/api/models/verify`, `/api/models/select`, `DELETE /api/models/{model_id}`, `/api/models/{model_id}/test`, `/api/system`, `/api/system/gpu`, а также `/health/live`, `/health/ready`, `/health/models`, `/health/gpu`.

## Snap Docker не поддерживается для GPU

`preflight.sh gpu` останавливает установку при Snap Docker или `Docker Root Dir: /var/snap/docker/...`. Используйте Docker Engine из apt-репозитория Docker или системный `docker.io` и Docker Compose v2. После миграции проверьте `Docker Root Dir: /var/lib/docker` и сокет `/run/docker.sock`.

## Диагностика остановки контейнеров

При `cannot stop container: permission denied` проверьте одновременный Snap/system Docker, `which dockerd`, `which nvidia-container-runtime`, Docker Root Dir, AppArmor, `docker.socket` и старые контейнеры от другого daemon. Не используйте `kill -9` как штатный способ. Проверочный сценарий:

```bash
docker compose stop
docker compose start
docker compose restart
docker compose down
```

## Права и .env

Контейнер запускается не от root и поддерживает `HOST_UID`/`HOST_GID`. После изменения `.env` пересоздайте контейнер:

```bash
docker compose up -d --force-recreate
```

Production нельзя запускать с `GIGASCRIBE_SECRET_KEY=change-me-long-random-string`; сгенерируйте ключ через `openssl rand -hex 32`.

## Тесты

```bash
python -m pytest
python -m pytest tests/test_requirements_integrity.py tests/test_password_validation.py
scripts/preflight.sh cpu
```

## Migration guide

1. Сделайте backup `data/` и `models/`.
2. Разделите старый `requirements.txt` на новые файлы или используйте поставляемые файлы.
3. Создайте `.env` с безопасным `GIGASCRIBE_SECRET_KEY`, `HOST_UID`, `HOST_GID`.
4. Для GPU удалите Snap Docker и установите Docker/Compose v2 + NVIDIA Container Toolkit.
5. Перескачайте/проверьте модели через `scripts/download_models.py --offline`; при ошибке выполните загрузку с `--force`.
6. Запустите CPU или GPU compose-команду и проверьте `/health/ready`.

## Известные ограничения

В этом репозитории нет доступа к реальной RTX 5060 Ti, поэтому фактические замеры VRAM и скорости должны выполняться на целевом хосте через `/api/system/gpu` и job logs. Реальный CUDA-smoke, pyannote gated downloads и инференс зависят от установленного драйвера, HF-доступа и локальных model snapshots.
