from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from trading_agent.monitoring import healthcheck


def test_ready_exits_successfully() -> None:
    with patch.object(healthcheck, "urlopen", return_value=BytesIO(b'{"status":"ready"}')):
        assert healthcheck.main() == 0


@pytest.mark.parametrize(
    "error",
    [
        HTTPError("http://127.0.0.1:8000/ready", 503, "Unavailable", {}, None),
        URLError("Connection refused"),
        TimeoutError("Timed out"),
    ],
)
def test_unavailable_exits_unsuccessfully(error: OSError) -> None:
    with patch.object(healthcheck, "urlopen", side_effect=error):
        assert healthcheck.main() == 1
