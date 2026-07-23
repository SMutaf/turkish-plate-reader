"""Çıkarım servisinin Pydantic şemaları (istek/cevap sözleşmesi).

KOORDİNAT DÜZLEMİ: Cevaptaki TÜM `box` alanları [x1, y1, x2, y2] piksel
cinsindendir ve sunucunun gerçekten işlediği JPEG'in `frame_w` × `frame_h`
düzlemindedir. İstemci, kendi ekran/kaynak çözünürlüğü farklıysa kutuları
frame_w/frame_h'ye göre ölçeklemelidir.
"""
from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Araç sınıfı serbest string değil, sabit küme
VehicleClass = Literal["car", "motorcycle", "bus", "truck"]

# Kutu: tam 4 tamsayı [x1, y1, x2, y2], piksel
Box = Annotated[
    List[int],
    Field(min_length=4, max_length=4,
          description="[x1, y1, x2, y2] piksel; frame_w×frame_h düzleminde"),
]


class Plate(BaseModel):
    """Bir araca ait okunmuş plaka."""

    box: Box
    conf: float = Field(ge=0.0, le=1.0, description="Plaka tespit güveni (0.0–1.0)")
    raw: str = Field(description="OCR'ın ham çıktısı, temizlenmemiş")
    text: str = Field(
        description="format_plate() sonucu; valid=false olsa da en iyi tahmini döner")
    valid: bool = Field(description="Tüketicinin bakacağı TEK karar bayrağı")
    votes: int = Field(ge=0, description="Bu sonucu üreten geçerli okuma sayısı")


class Vehicle(BaseModel):
    """Tespit + takip edilen bir araç ve (varsa) plakası."""

    model_config = ConfigDict(populate_by_name=True)

    track_id: Optional[int] = Field(
        default=None,
        description="ByteTrack track kimliği (camera_id içinde tekil); "
                    "tracker atayamazsa null (araç yine listelenir, plate=null olur)")
    # 'class' Python anahtar sözcüğü; JSON'da "class" olarak çıkar (alias).
    vehicle_class: VehicleClass = Field(
        alias="class", description="Araç sınıfı (car/motorcycle/bus/truck)")
    box: Box
    conf: float = Field(ge=0.0, le=1.0, description="Araç tespit güveni (0.0–1.0)")
    plate: Optional[Plate] = Field(
        default=None, description="Araç okunamadıysa null")


class OrphanPlate(BaseModel):
    """Hiçbir araç kutusunun içinde olmayan plaka tespiti.

    NOT: Orphan plakalara OCR UYGULANMAZ (OCR track tabanlıdır); bu yüzden
    `text` daima null'dur. Boş string döndürülmez.
    """

    box: Box
    conf: float = Field(ge=0.0, le=1.0, description="Plaka tespit güveni (0.0–1.0)")
    text: Optional[str] = Field(
        default=None, description="Orphan plakalara OCR uygulanmaz; daima null")


class InferResponse(BaseModel):
    """POST /v1/infer başarılı cevabı.

    Tüm box koordinatları frame_w × frame_h piksel düzlemindedir (bkz. modül
    docstring'i).
    """

    camera_id: str
    frame_ts: int = Field(description="İstemcinin gönderdiği kare zaman damgası (unix ms)")
    latency_ms: int = Field(ge=0, description="Sunucunun bu kareyi işleme süresi (ms)")
    frame_w: int = Field(gt=0, description="İşlenen JPEG genişliği (px); tüm box'lar bu düzlemde")
    frame_h: int = Field(gt=0, description="İşlenen JPEG yüksekliği (px); tüm box'lar bu düzlemde")
    vehicles: List[Vehicle]
    orphan_plates: List[OrphanPlate]


class ErrorResponse(BaseModel):
    """Hata cevabı.

    Kullanılacak HTTP kodları:
    - 400: bozuk/çözülemeyen JPEG veya eksik/hatalı form alanı (camera_id, frame_ts)
    - 500: model yükleme / çıkarım sırasında beklenmeyen hata
    """

    error: str = Field(description="Kısa hata türü/mesajı")
    detail: Optional[str] = Field(default=None, description="Ek ayrıntı; yoksa null")


class HealthResponse(BaseModel):
    """GET /health cevabı."""

    # 'model_loaded' pydantic'in korumalı 'model_' ad alanıyla çakışır; kapat.
    model_config = ConfigDict(protected_namespaces=())

    model_loaded: bool = Field(description="Modeller belleğe yüklendi ve warmup yapıldı mı")
    device: str = Field(description="Çıkarım cihazı: 'cuda' veya 'cpu'")
    gpu_memory_reserved_mb: Optional[float] = Field(
        default=None,
        description="torch.cuda.memory_reserved() (MB); cihaz cpu ise null")
    uptime_sec: float = Field(ge=0.0, description="Sunucunun ayakta olduğu süre (sn)")
    active_cameras: int = Field(ge=0, description="Model yüklenmiş aktif kamera sayısı")
    camera_ids: List[str] = Field(default_factory=list,
                                  description="Aktif kamera kimlikleri")
