# AI PDF Analyzer (RAG & Celery)

![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.14-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688)

Интеллектуальная система анализа PDF-документов. Позволяет загружать файлы, автоматически разбивать их на части, генерировать эмбеддинги через Ollama и отвечать на вопросы по контексту (RAG).

## Основные технологии
* **Backend:** FastAPI, Celery
* **AI Stack:** LangChain, Ollama
* **Database:** PostgreSQL + **pgvector**
* **Cache/Broker:** Redis
* **Testing:** Pytest + Manual Monkeypatching

## Архитектура
1.  **Ingestion:** PDF -> Чанки (RecursiveCharacterTextSplitter) -> Векторизация (Ollama) -> Стор в PGVector.
2.  **RAG Pipeline:** Вопрос пользователя -> Эмбеддинг вопроса -> Сходство косинусов в БД -> Контекст -> LLM ответ.

## Инструкция по запуску

Для работы приложения требуются **Docker** и **Docker Compose**.

---

## 1. Подготовка окружения
Создайте файл `.env` на основе примера:
```bash
cp .env.example .env
```
Убедитесь, что в .env указаны корректные URL для PostgreSQL, Redis и Ollama.

## 2. Запуск инфраструктуры
Соберите и запустите контейнеры:
```bash
docker-compose up --build -d
```
Эта команда поднимет:

1. PostgreSQL (pgvector) — база данных для хранения векторов.
2. Redis — брокер сообщений для Celery.
3. App — FastAPI сервер (доступен по адресу http://localhost:8000).
4. Worker — фоновый процесс для обработки тяжелых PDF-задач.

## 3. Настройка Ollama
Если вы используете Ollama локально, убедитесь, что она доступна для Docker-контейнеров, и модели загружены:
```bash
ollama pull mxbai-embed-large
ollama pull llama3.1
```

## Тестирование
Локальный запуск (требуется venv):
```bash
pytest -v --cov=app
```
Запуск внутри Docker-контейнера:
```bash
docker exec -it <имя_контейнера_app> pytest -v --cov=app
```
