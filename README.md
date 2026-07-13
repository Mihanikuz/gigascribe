# GigaScribe Local Server

Локальная web-версия для Ubuntu 24.04. Модели и результаты хранятся на сервере; внешняя публикация Gradio отключена.

## Запуск

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y ffmpeg
export GIGASCRIBE_SECRET_KEY="change-me"
export GIGASCRIBE_ADMIN_PASSWORD="change-me"
export GIGASCRIBE_DATA_DIR="/opt/gigascribe/data"
export GIGASCRIBE_GIGAAM_MODEL="/opt/gigascribe/models/gigaam-v2-ctc"
export GIGASCRIBE_PYANNOTE_MODEL="/opt/gigascribe/models/pyannote-speaker-diarization-3.1"
uvicorn server:app --host 0.0.0.0 --port 8000
```

Если `GIGASCRIBE_GIGAAM_MODEL` или `GIGASCRIBE_PYANNOTE_MODEL` не заданы, используются прежние имена моделей библиотек. Для полностью автономной установки укажите локальные директории моделей.

## Пользователи и AD

При первом запуске создается локальный администратор `admin` с паролем из `GIGASCRIBE_ADMIN_PASSWORD` (по умолчанию `admin`). Локальные пользователи хранятся в JSON-файле `GIGASCRIBE_USERS_FILE` или в `$GIGASCRIBE_DATA_DIR/users.json`.

Создание локального пользователя:

```bash
curl -u admin:change-me -F username=ivan -F password='strong-password' http://localhost:8000/admin/users
```

Для Active Directory/LDAP задайте:

```bash
export GIGASCRIBE_LDAP_SERVER="ldaps://dc.example.local:636"
export GIGASCRIBE_AD_DOMAIN="EXAMPLE"
```

## Возможности

- Авторизация по локальному логину/паролю или через AD bind.
- Многопользовательский web-режим: каждый пользователь видит только свои задания.
- Асинхронная обработка с индикатором прогресса.
- Скачивание транскрипта, лога задания, исходного аудио и нормализованного WAV.
