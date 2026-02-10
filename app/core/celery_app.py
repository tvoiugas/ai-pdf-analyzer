from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

celery_app = Celery(
    "worker",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    include=["app.services.pdf_service", "app.services.ask_service"]
)

celery_app.conf.task_routes = {
    "app.services.pdf_service.pdf_task": "main-queue"
    # "app.services.ask_service.get_answer": {"queue": "ask_tasks"},
}

celery_app.conf.update(
    broker_transport_options={
        'visibility_timeout': 3600,
    },
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
