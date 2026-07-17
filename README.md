# GigaScribe

Локальный сервис транскрибации для Ubuntu/Docker Compose с режимами CPU и NVIDIA GPU.

## Поддерживаемые модели

ASR: `GigaAM v2_ctc`, `GigaAM v3_ctc`, `GigaAM v3_e2e_rnnt`. Диаризация: `pyannote speaker-diarization-3.1`, `pyannote Community-1`, `Без диаризации`. Выбор сохраняется в `models/settings.json`; новые задания получают снимок настроек, уже запущенные задания не переключаются молча.

### Совместимость моделей

GigaAM устанавливается из официального репозитория `salute-developers/GigaAM` по
зафиксированному release `v0.3.0` (см. `requirements-gigaam.txt`), а не из
устаревшего `gigaam==0.1.0`. Официальные имена, передаваемые в
`gigaam.load_model`, — `v2_ctc`, `v3_ctc` и `v3_e2e_rnnt`.

CPU использует `torch/torchaudio 2.7.0+cpu`; GPU — официальный
`2.7.0+cu128`. Маркер модели создаётся загрузчиком только после конструирования
модели в изолированном кэше; он содержит SHA-256 checkpoint, release backend и
версию PyTorch. Смена release или PyTorch делает старый marker неготовым.

`speaker-diarization-3.1` — legacy pipeline (`pyannote.audio 3.1.*`), тогда как
Community-1 использует современный API (`pyannote.audio 4.*`). Их нельзя
считать взаимозаменяемыми: production deployment должен предоставлять legacy
runtime отдельно, пока не пройдёт offline load-test именно этой pipeline.
Community-1 не скачивает `segmentation-3.0`; загрузчик получает только
зависимости, объявленные metadata выбранной pipeline.

Запускайте ровно один профиль:

```bash
docker compose -f compose.yaml --profile cpu up -d
# или
docker compose -f compose.yaml --profile gpu up -d
```

## Быстрый старт CPU

```bash
cp .env.example .env  # если шаблон отсутствует, создайте .env вручную
printf 'GIGASCRIBE_SECRET_KEY=%s\nGIGASCRIBE_ADMIN_PASSWORD=%s\nHOST_UID=%s\nHOST_GID=%s\n' "$(openssl rand -hex 32)" "change-this-password" "$(id -u)" "$(id -g)" > .env
scripts/preflight.sh cpu
docker compose up -d --build
```

CPU-образ использует `requirements-cpu.txt` и не скачивает CUDA wheels.

## Устойчивость сборки к обрывам сети

Dockerfile использует BuildKit cache mount с постоянным идентификатором `gigascribe-pip-cache` для `/root/.cache/pip`. Этот каталог управляется builder'ом Docker, а не находится внутри финального образа: он сохраняет HTTP-кэш и полностью скачанные wheels между запусками `docker compose build` на том же builder'е. Поэтому уже завершённые загрузки, включая CPU wheels `torch` и `torchaudio`, при неизменных версиях берутся из кэша, а не загружаются повторно.

Каждая установка зависимостей использует таймаут 600 секунд и 20 попыток сетевого подключения. Неподдерживаемая pip-опция `--resume-retries` намеренно не используется. Завершённые wheels сохраняются в BuildKit cache; если сборка прервана до завершения конкретного wheel, pip скачает только этот незавершённый wheel заново.

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

GPU-сборка использует `requirements-cu128.txt` с жёсткими пинами `torch==2.7.0+cu128`, `torchaudio==2.7.0+cu128` и индексом `https://download.pytorch.org/whl/cu128`. Суффикс `+cu128` обязателен: он не даёт pip незаметно выбрать CPU, cu121 или cu124 wheel. При старте GPU-контейнера автоматически выполняется CUDA smoke-test; для RTX 5060 Ti он требует CUDA 12.8 и наличие `sm_120`.

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

`requirements-gigaam.txt` закрепляет совместимые runtime-зависимости GigaAM (`onnx`, `onnxruntime`, `hydra-core`, `omegaconf`, `sentencepiece`). Затем `gigaam==0.1.0` устанавливается ровно один раз через `pip install --no-deps`, потому что metadata пакета ограничивает `torch<=2.5.1`. Это сохраняет уже установленный CUDA 12.8 PyTorch и исключает повторную установку `torch`/`torchaudio`.

Для локального pyannote используется путь к `config.yaml`, marker `.pyannote-ready.json` создаётся только после загрузочной проверки. HF token лучше передавать одноразово или через secret; не публикуйте вывод `docker compose config`, он может раскрыть `.env`.

## API и UI

Доступны `/api/models`, `/api/models/status`, `/api/models/download`, `/api/models/verify`, `/api/models/select`, `DELETE /api/models/{model_id}`, `/api/models/{model_id}/test`, `/api/system`, `/api/system/gpu`, а также `/health/live`, `/health/ready`, `/health/models`, `/health/gpu`. `/health/models` выполняет загрузочную проверку и возвращает отдельные статусы `import`, `dependencies`, `checkpoint`, `load` и `device`; `/api/models/status` показывает путь, размер, готовность к загрузке и состояние тестового inference для каждой модели.

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

Контейнер при каждом старте создаёт `data/`, `data/uploads/` и `models/`, исправляет их владельца на `HOST_UID`/`HOST_GID`, затем запускает приложение с этими правами. После изменения `.env` пересоздайте контейнер:

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

## Compose

`compose.yaml` is the sole canonical Compose definition. Use `docker compose -f compose.yaml ...`; the legacy `docker-compose.yml` was removed to avoid Compose selecting an ambiguous configuration.
