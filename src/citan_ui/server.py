"""Small dependency-free HTTP server for the local CitAn application."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

from .service import AnalysisService


LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 100_000

STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class CitAnRequestHandler(BaseHTTPRequestHandler):
    """Serve the frontend and the JSON inference API."""

    server_version = "CitAnUI/1.0"
    service: ClassVar[AnalysisService]
    frontend_dir: ClassVar[Path]

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 (method name defined by stdlib)
        static = STATIC_ROUTES.get(self.path)
        if static is not None:
            filename, content_type = static
            self._send(
                HTTPStatus.OK,
                (self.frontend_dir / filename).read_bytes(),
                content_type,
            )
            return
        if self.path == "/api/health":
            self._send_json(HTTPStatus.OK, self.service.health())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802 (method name defined by stdlib)
        if self.path != "/api/analyze":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            content_type = self.headers.get("Content-Type", "")
            if "application/json" not in content_type.lower():
                raise ValueError("Content-Type must be application/json.")

            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("Invalid request size.")

            request = json.loads(self.rfile.read(length))
            result = self.service.analyze(
                str(request.get("citance", "")),
                str(request.get("model", "scibert")),
            )
            self._send_json(HTTPStatus.OK, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # Keep tracebacks in server logs, not API responses.
            LOGGER.exception("Citance analysis failed")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(exc).__name__}: {exc}"},
            )

    def log_message(self, format_string: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)


def create_server(
    host: str,
    port: int,
    *,
    service: AnalysisService | None = None,
) -> ThreadingHTTPServer:
    """Build a configured HTTP server without starting its event loop."""

    analysis_service = service or AnalysisService()
    analysis_service.settings.validate_artifacts()
    CitAnRequestHandler.service = analysis_service
    CitAnRequestHandler.frontend_dir = analysis_service.settings.frontend_dir
    return ThreadingHTTPServer((host, port), CitAnRequestHandler)
