import re
from unittest.mock import patch

import pytest

from app import main as main_module
from app.main import display_filename

PDF_BYTES = b"%PDF-1.4\n%fake but well-formed enough for the header check\n"
STORED_NAME = re.compile(r"[0-9a-f]{32}\.pdf")


@pytest.fixture
def delay():
    with patch("app.services.pdf_service.process_pdf_task.delay") as mock_delay:
        yield mock_delay


async def upload(client, name, content=PDF_BYTES):
    return await client.post("/upload", files={"file": (name, content, "application/pdf")})


@pytest.mark.parametrize("raw_name,expected", [
    ("report.pdf", "report.pdf"),
    ("../../etc/cron.d/evil.pdf", "evil.pdf"),
    (r"..\..\evil.pdf", "evil.pdf"),
    ("/etc/evil.pdf", "evil.pdf"),
    ("bad\x00\x1b[31mname.pdf", "bad[31mname.pdf"),
    ("  spaced.pdf  ", "spaced.pdf"),
    ("x" * 300 + ".pdf", "x" * 251 + ".pdf"),
    ("../", ""),
    (None, ""),
])
def test_display_filename(raw_name, expected):
    assert display_filename(raw_name) == expected


@pytest.mark.parametrize("name", [
    "../../etc/cron.d/evil.pdf",
    r"..\..\evil.pdf",
    "/etc/evil.pdf",
    "x" * 300 + ".pdf",
])
async def test_client_filename_never_reaches_the_path(client, as_user, upload_dir, delay, name):
    as_user(7)
    response = await upload(client, name)

    assert response.status_code == 200
    stored = list(upload_dir.iterdir())
    assert len(stored) == 1 and STORED_NAME.fullmatch(stored[0].name)
    saved_path, label, owner = delay.call_args.args
    assert saved_path == str(stored[0])
    assert label == display_filename(name) and "/" not in label and "\\" not in label
    assert owner == 7


async def test_same_name_twice_gets_separate_files(client, as_user, upload_dir, delay):
    as_user()
    await upload(client, "report.pdf")
    await upload(client, "report.pdf")

    stored = sorted(p.name for p in upload_dir.iterdir())
    assert len(stored) == 2
    assert delay.call_args_list[0].args[0] != delay.call_args_list[1].args[0]


async def test_extension_check_ignores_case(client, as_user, upload_dir, delay):
    as_user()
    response = await upload(client, "SCAN.PDF")

    assert response.status_code == 200
    assert response.json()["filename"] == "SCAN.PDF"


@pytest.mark.parametrize("name,content", [
    pytest.param("notes.txt", PDF_BYTES, id="wrong-extension"),
    pytest.param("../", PDF_BYTES, id="no-filename"),
    pytest.param("fake.pdf", b"MZ\x90\x00 definitely an executable", id="not-a-pdf"),
    pytest.param("empty.pdf", b"", id="empty"),
])
async def test_rejected_uploads_leave_nothing_behind(client, as_user, upload_dir, delay, name, content):
    as_user()
    response = await upload(client, name, content)

    assert response.status_code == 400
    assert list(upload_dir.iterdir()) == []
    assert not delay.called


async def test_oversized_upload_is_rejected_and_removed(client, as_user, upload_dir, delay, monkeypatch):
    as_user()
    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 2 * main_module.UPLOAD_CHUNK_SIZE)
    # Big enough that the limit is hit mid-write, after the first chunk is already on disk.
    response = await upload(client, "big.pdf", PDF_BYTES + b"0" * (3 * main_module.UPLOAD_CHUNK_SIZE))

    assert response.status_code == 413
    assert list(upload_dir.iterdir()) == []
    assert not delay.called


async def test_broker_failure_removes_the_saved_file(client, as_user, upload_dir, delay):
    as_user()
    delay.side_effect = ConnectionError("Redis is down")

    with pytest.raises(ConnectionError):
        await upload(client, "report.pdf")

    assert list(upload_dir.iterdir()) == []
