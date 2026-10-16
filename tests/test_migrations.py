from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.core.celery_app import celery_app

ALEMBIC_INI = Path(__file__).resolve().parent.parent / "alembic.ini"


def test_migration_history_is_a_single_line():
    heads = ScriptDirectory.from_config(Config(str(ALEMBIC_INI))).get_heads()

    assert len(heads) == 1


@pytest.mark.db
def test_models_match_the_migrations():
    """Fails when a model changes without a migration (the database must be at `alembic upgrade head`)."""
    command.check(Config(str(ALEMBIC_INI)))


def test_celery_routes_point_at_registered_tasks():
    celery_app.loader.import_default_modules()
    routes = celery_app.conf.task_routes or {}

    assert "app.services.pdf_service.process_pdf_task" in celery_app.tasks
    assert all(name in celery_app.tasks for name in routes)
