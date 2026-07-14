# GigaScribe Local Server

Локальный многопользовательский web-сервер для транскрибации. Приложение запускается в Docker, данные и модели вынесены в отдельные каталоги на хосте:

- `./data` — пользователи, загруженные файлы и результаты;
- `./models` — кэши и локальные snapshots моделей (`HF_HOME`, `TORCH_HOME`, pyannote).

## Требования

- Docker и Docker Compose;
- `HF_TOKEN` для модели `pyannote/speaker-diarization-3.1`: предварительно примите условия доступа к модели на Hugging Face и создайте токен;
- для GPU — установленный NVIDIA Container Toolkit и раскомментированный GPU-блок в `docker-compose.yml`.

## Быстрый запуск в Docker

1. Подготовьте конфигурацию и каталоги:

   ```bash
   cp .env.example .env
   mkdir -p data models
   ```

2. Отредактируйте `.env`:

   ```dotenv
   GIGASCRIBE_SECRET_KEY=change-me-long-random-string
   GIGASCRIBE_ADMIN_PASSWORD=change-me
   HF_TOKEN=hf_...
   GIGASCRIBE_PYANNOTE_MODEL=/opt/gigascribe/models/pyannote-speaker-diarization-3.1
   GIGASCRIBE_GIGAAM_MODEL=v2_ctc
   ```

3. Соберите образ:

   ```bash
   docker compose build
   ```

4. Скачайте модели в отдельный каталог `./models`:

   ```bash
   docker compose run --rm gigascribe python scripts/download_models.py --models-dir /opt/gigascribe/models
   ```

   Скрипт прогревает GigaAM и скачивает snapshot `pyannote/speaker-diarization-3.1`. Все файлы сохраняются в volume `./models:/opt/gigascribe/models`.

5. Запустите сервер:

   ```bash
   docker compose up -d
   ```

6. Откройте приложение: <http://localhost:8000>. При первом запуске создается локальный администратор `admin` с паролем из `GIGASCRIBE_ADMIN_PASSWORD`.

## Обновление или повторное скачивание моделей

```bash
docker compose run --rm gigascribe python scripts/download_models.py --models-dir /opt/gigascribe/models
```

Полезные варианты:

```bash
# Скачать только pyannote, не прогревая GigaAM
docker compose run --rm gigascribe python scripts/download_models.py --skip-gigaam

# Прогреть только GigaAM, не скачивая pyannote
docker compose run --rm gigascribe python scripts/download_models.py --skip-pyannote
```

Если модель pyannote уже скачана вручную, укажите ее локальный каталог в `.env`:

```dotenv
GIGASCRIBE_PYANNOTE_MODEL=/opt/gigascribe/models/pyannote-speaker-diarization-3.1
```

## Переменные окружения

| Переменная | Назначение | Значение в Docker |
| --- | --- | --- |
| `GIGASCRIBE_DATA_DIR` | каталог данных приложения | `/opt/gigascribe/data` |
| `GIGASCRIBE_MODELS_DIR` | отдельный каталог моделей | `/opt/gigascribe/models` |
| `HF_HOME` | кэш Hugging Face | `/opt/gigascribe/models/huggingface` |
| `TORCH_HOME` | кэш Torch | `/opt/gigascribe/models/torch` |
| `GIGASCRIBE_GIGAAM_MODEL` | имя или путь модели GigaAM | `v2_ctc` |
| `GIGASCRIBE_PYANNOTE_MODEL` | локальный путь или repo id pyannote | `/opt/gigascribe/models/pyannote-speaker-diarization-3.1` |
| `HF_TOKEN` | токен Hugging Face для pyannote | задается в `.env` |
| `GIGASCRIBE_SECRET_KEY` | ключ сессий | задается в `.env` |
| `GIGASCRIBE_ADMIN_PASSWORD` | пароль первого локального admin | задается в `.env` |

## Управление пользователями и AD

Создание локального пользователя:

```bash
curl -u admin:change-me -F username=ivan -F password='strong-password' http://localhost:8000/admin/users
```

Для Active Directory/LDAP добавьте в `.env`:

```dotenv
GIGASCRIBE_LDAP_SERVER=ldaps://dc.example.local:636
GIGASCRIBE_AD_DOMAIN=EXAMPLE
```

## Локальный запуск без Docker

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y ffmpeg
export GIGASCRIBE_SECRET_KEY="change-me"
export GIGASCRIBE_ADMIN_PASSWORD="change-me"
export GIGASCRIBE_DATA_DIR="$PWD/data"
export GIGASCRIBE_MODELS_DIR="$PWD/models"
export HF_HOME="$PWD/models/huggingface"
export TORCH_HOME="$PWD/models/torch"
export GIGASCRIBE_PYANNOTE_MODEL="$PWD/models/pyannote-speaker-diarization-3.1"
python scripts/download_models.py --models-dir "$PWD/models"
uvicorn server:app --host 0.0.0.0 --port 8000
```
