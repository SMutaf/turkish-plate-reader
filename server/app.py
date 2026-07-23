"""FastAPI çıkarım servisi (GPU'lu makinede çalışır).

ÇALIŞTIRMA — repo kökünden, TEK worker ile:
    uvicorn server.app:app --host 127.0.0.1 --port 8000 --workers 1
    # veya CLI önceliğiyle:
    python -m server.app --port 8000 --max-cameras 4

UYARI: --workers 1 ZORUNLU. Çoklu worker model belleğini kopyalar; 4 GB
kartta (GTX 1650) belleği taşırır.

EŞZAMANLILIK: GPU tek kaynaktır. Çıkarım bir asyncio.Lock ile serileştirilir,
ama blocking çağrı run_in_executor ile threadpool'a atılır — böylece lock
tutulurken bile event loop bloklanmaz ve /health anında cevap verir.
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from server.inference import InferenceConfig, InferenceEngine
from server.schemas import ErrorResponse, HealthResponse, InferResponse

CONFIG_PATH = Path("config.yaml")
ENV_PREFIX = "PLATE_"

# `python -m server.app` ile verilen CLI değerleri (en yüksek öncelik)
_CLI_OVERRIDES: Dict[str, Any] = {}


@dataclass
class ServerConfig:
    """Sunucu (HTTP) ayarları; çıkarım ayarları InferenceConfig'te."""

    host: str = "127.0.0.1"
    port: int = 8000
    max_upload_mb: float = 10.0


_INFER_FIELDS: Dict[str, type] = {
    "vehicle_conf": float, "plate_conf": float, "ocr_min_conf": float,
    "vote_window": int, "track_max_age": float, "max_cameras": int,
    "max_ocr_per_frame": int, "vehicle_weights": str, "plate_weights": str,
}
_SERVER_FIELDS: Dict[str, type] = {"host": str, "port": int, "max_upload_mb": float}


def _yaml_server_section() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    import yaml
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return data.get("server", {}) or {}


def _env_section(fields: Dict[str, type]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in fields:
        val = os.environ.get(ENV_PREFIX + key.upper())
        if val is not None:
            out[key] = val
    return out


def _overlay(target: Any, fields: Dict[str, type], source: Dict[str, Any]) -> None:
    for key, typ in fields.items():
        if source.get(key) is not None:
            setattr(target, key, typ(source[key]))


def load_config() -> Tuple[InferenceConfig, ServerConfig]:
    """Öncelik: CLI > env (PLATE_*) > config.yaml > varsayılan."""
    infer_cfg, srv_cfg = InferenceConfig(), ServerConfig()
    yaml_cfg = _yaml_server_section()
    for source in (yaml_cfg, _env_section(_INFER_FIELDS), _CLI_OVERRIDES):
        _overlay(infer_cfg, _INFER_FIELDS, source)
    for source in (yaml_cfg, _env_section(_SERVER_FIELDS), _CLI_OVERRIDES):
        _overlay(srv_cfg, _SERVER_FIELDS, source)
    return infer_cfg, srv_cfg


ENGINE: Optional[InferenceEngine] = None
START_TS: float = 0.0
_LOCK = asyncio.Lock()   # GPU tek kaynak: çıkarımları serileştirir


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Başlangıçta modelleri yükle + warmup yap. Ağırlık yoksa BAŞLATMA."""
    global ENGINE, START_TS
    infer_cfg, srv_cfg = load_config()
    app.state.server_config = srv_cfg

    for path in (infer_cfg.plate_weights, infer_cfg.vehicle_weights):
        if not Path(path).exists():
            raise RuntimeError(
                f"Model dosyası bulunamadı: {path!r} — sunucu BAŞLATILMIYOR. "
                "Ağırlıkları yerleştirin veya config'te yolu düzeltin.")

    ENGINE = InferenceEngine(infer_cfg)
    ENGINE.warmup()          # ilk isteği bekletme; GPU belleği burada ölçülür
    START_TS = time.time()
    print(f"[app] hazır | device={ENGINE.device()} "
          f"| max_cameras={infer_cfg.max_cameras} "
          f"| max_upload_mb={srv_cfg.max_upload_mb}", flush=True)
    yield


app = FastAPI(title="Türk Plaka Çıkarım Servisi", version="1.0.0", lifespan=lifespan)


# --- Hata cevapları: HER ZAMAN ErrorResponse şeması (FastAPI'nin {"detail":...} değil)
_ERROR_NAMES = {400: "bad_request", 413: "payload_too_large",
                503: "service_unavailable", 500: "internal_error"}


def _error_json(status_code: int, detail: Optional[str]) -> JSONResponse:
    body = ErrorResponse(error=_ERROR_NAMES.get(status_code, "http_error"),
                         detail=detail)
    return JSONResponse(status_code=status_code, content=body.model_dump())


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return _error_json(exc.status_code, str(exc.detail) if exc.detail else None)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # eksik/hatalı form alanı -> 400 (FastAPI varsayılanı 422)
    missing = [".".join(str(p) for p in e.get("loc", [])[1:]) for e in exc.errors()]
    return _error_json(400, f"Eksik/hatalı alan: {', '.join(missing) or 'bilinmiyor'}")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return _error_json(500, f"{type(exc).__name__}: {exc}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Sağlık durumu. LOCK BEKLEMEZ — çıkarım sürerken de anında cevap verir."""
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Motor henüz hazır değil")
    cams = ENGINE.active_cameras()
    return HealthResponse(
        model_loaded=ENGINE.models_loaded(),
        device=ENGINE.device(),
        gpu_memory_reserved_mb=ENGINE.gpu_mem_reserved_mb(),
        uptime_sec=round(time.time() - START_TS, 3),
        active_cameras=len(cams),
        camera_ids=cams,
    )


@app.post("/v1/infer", response_model=InferResponse)
async def infer_endpoint(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    frame_ts: int = Form(...),
):
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Motor henüz hazır değil")

    srv: ServerConfig = app.state.server_config
    data = await file.read()

    limit = int(srv.max_upload_mb * 1024 * 1024)
    if len(data) > limit:
        raise HTTPException(status_code=413,
                            detail=f"Dosya çok büyük: {len(data)} bayt > {limit} bayt "
                                   f"(max_upload_mb={srv.max_upload_mb})")
    if not data:
        raise HTTPException(status_code=400, detail="Boş dosya gönderildi")

    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400,
                            detail="Görüntü çözülemedi (geçerli bir JPEG değil)")

    cams = ENGINE.active_cameras()
    if camera_id not in cams and len(cams) >= ENGINE.cfg.max_cameras:
        raise HTTPException(
            status_code=503,
            detail=f"max_cameras={ENGINE.cfg.max_cameras} dolu; '{camera_id}' "
                   f"kabul edilmedi. Aktif kameralar: {cams}")

    # GPU tek kaynak: lock ile serileştir, ama blocking çağrıyı executor'a at
    loop = asyncio.get_running_loop()
    async with _LOCK:
        result = await loop.run_in_executor(
            None, ENGINE.infer, frame, camera_id, frame_ts)
    return result


def main() -> None:
    """CLI önceliğiyle çalıştırma: python -m server.app --port 8000 ..."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Plaka çıkarım servisi")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--max-upload-mb", type=float, dest="max_upload_mb")
    parser.add_argument("--max-cameras", type=int, dest="max_cameras")
    parser.add_argument("--vehicle-conf", type=float, dest="vehicle_conf")
    parser.add_argument("--plate-conf", type=float, dest="plate_conf")
    parser.add_argument("--vote-window", type=int, dest="vote_window")
    parser.add_argument("--max-ocr-per-frame", type=int, dest="max_ocr_per_frame")
    args = parser.parse_args()

    _CLI_OVERRIDES.update({k: v for k, v in vars(args).items() if v is not None})
    _, srv_cfg = load_config()
    # workers=1 ZORUNLU (model belleği kopyalanmasın)
    uvicorn.run(app, host=srv_cfg.host, port=srv_cfg.port, workers=1)


if __name__ == "__main__":
    main()
