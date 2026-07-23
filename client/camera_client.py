"""İnce kamera istemcisi — MODEL YOK, GPU YOK, ağır bağımlılık YOK.

Kareyi JPEG olarak sunucudaki /v1/infer'e gönderir, dönen JSON'u çizer.
Bağımlılıklar: yalnızca opencv-python + requests (bkz. client/requirements.txt).

TEMEL KURAL: aynı anda TEK istek. Cevap beklenirken gelen kareler ATILIR,
kuyruğa alınmaz — böylece gecikme birikmez.

Ayarlar komut satırından alınır (istemcide yaml bağımlılığı yoktur):

    python client/camera_client.py --server http://127.0.0.1:8000 \
        --camera-id kapi1 --source 0 --fps 5 --jpeg-quality 85

  --source : webcam indeksi (0,1,...) | video dosyası | rtsp://... URL
  --headless : GUI olmadan konsola yaz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import requests

# Renkler (BGR)
COLOR_VEHICLE = (0, 200, 255)    # turuncu/sarı: araç
COLOR_PLATE_VALID = (0, 220, 0)  # yeşil: geçerli plaka
COLOR_PLATE_WEAK = (0, 165, 255) # turuncu: okundu ama valid değil
COLOR_WARN = (0, 0, 255)         # kırmızı: bağlantı yok

BACKOFF_START = 1.0
BACKOFF_MAX = 30.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plaka okuma kamera istemcisi (ince)")
    p.add_argument("--server", default="http://127.0.0.1:8000",
                   help="Çıkarım sunucusu adresi")
    p.add_argument("--camera-id", default="kapi1", help="Bu kameranın kimliği")
    p.add_argument("--source", default="0",
                   help="webcam indeksi (0,1,...) | video dosyası | rtsp:// URL")
    p.add_argument("--fps", type=float, default=5.0, dest="target_fps",
                   help="Saniyede gönderilecek kare sayısı (varsayılan 5)")
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="JPEG kalitesi (varsayılan 85)")
    p.add_argument("--timeout", type=float, default=10.0,
                   help="HTTP zaman aşımı (sn)")
    p.add_argument("--headless", action="store_true",
                   help="Pencere açma, konsola yaz")
    return p.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    """webcam indeksi / video dosyası / rtsp:// için VideoCapture açar."""
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    if source.lower().startswith("rtsp://"):
        # RTSP: FFMPEG arka ucu + TCP taşıma (UDP paket kaybına karşı)
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)


def post_frame(session: requests.Session, url: str, frame, camera_id: str,
               jpeg_quality: int, timeout: float):
    """Kareyi JPEG olarak gönderir. (cevap_json, rtt_ms) döndürür."""
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG kodlanamadı")
    frame_ts = int(time.time() * 1000)
    t0 = time.time()
    r = session.post(
        f"{url}/v1/infer",
        files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
        data={"camera_id": camera_id, "frame_ts": frame_ts},
        timeout=timeout,
    )
    rtt_ms = (time.time() - t0) * 1000
    r.raise_for_status()
    return r.json(), rtt_ms


def draw_overlay(frame, resp: dict, fps: float, rtt_ms: float, dropped: int) -> None:
    """Sunucudan dönen JSON'u kare üzerine çizer.

    NOT: kutular sunucunun işlediği frame_w×frame_h düzlemindedir; istemci
    kareyi yeniden boyutlandırmadan gönderdiği için ölçek 1:1'dir. Yine de
    güvenli olsun diye ölçek hesaplanır.
    """
    fh, fw = frame.shape[:2]
    sx = fw / max(resp.get("frame_w") or fw, 1)
    sy = fh / max(resp.get("frame_h") or fh, 1)

    def S(box):
        return [int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)]

    for v in resp.get("vehicles", []):
        x1, y1, x2, y2 = S(v["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VEHICLE, 2)
        tid = v.get("track_id")
        label = f"#{tid if tid is not None else '-'} {v['class']} {v['conf']:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_VEHICLE, 2)

        plate = v.get("plate")
        if plate:
            px1, py1, px2, py2 = S(plate["box"])
            color = COLOR_PLATE_VALID if plate["valid"] else COLOR_PLATE_WEAK
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            text = plate["text"] or "?"
            cv2.putText(frame, f"{text} ({plate['votes']})", (px1, max(py1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for op in resp.get("orphan_plates", []):
        x1, y1, x2, y2 = S(op["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PLATE_WEAK, 1)

    cv2.putText(frame, f"FPS {fps:4.1f} | RTT {rtt_ms:5.0f}ms | atilan {dropped}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def summarize(resp: dict) -> str:
    """Headless mod için kısa özet."""
    parts = []
    for v in resp.get("vehicles", []):
        p = v.get("plate")
        if p and p.get("text"):
            flag = "OK " if p["valid"] else "?? "
            parts.append(f"{flag}#{v.get('track_id')} {p['text']} (votes={p['votes']})")
    if not parts:
        return f"{len(resp.get('vehicles', []))} arac, plaka okunmadi"
    return " | ".join(parts)


def main() -> None:
    args = parse_args()
    cap = open_capture(args.source)
    if not cap.isOpened():
        sys.exit(f"HATA: görüntü kaynağı açılamadı: {args.source!r}")

    session = requests.Session()
    min_interval = 1.0 / max(args.target_fps, 0.1)
    last_sent = 0.0
    dropped = 0
    fps = 0.0
    backoff = BACKOFF_START
    connected = True

    print(f"[client] kaynak={args.source} sunucu={args.server} "
          f"camera_id={args.camera_id} hedef_fps={args.target_fps}", flush=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[client] görüntü kaynağı bitti/kapandı", flush=True)
                break

            now = time.time()
            # target_fps'e göre kare atla; cevap beklenirken biriken kareler de burada düşer
            if now - last_sent < min_interval:
                dropped += 1
                continue

            try:
                resp, rtt_ms = post_frame(session, args.server, frame,
                                          args.camera_id, args.jpeg_quality,
                                          args.timeout)
                if not connected:
                    print("[client] bağlantı geri geldi", flush=True)
                connected = True
                backoff = BACKOFF_START
                dt = time.time() - last_sent if last_sent else 0
                fps = (1.0 / dt) if dt > 0 else fps
                last_sent = time.time()

                if args.headless:
                    print(f"[{time.strftime('%H:%M:%S')}] RTT {rtt_ms:5.0f}ms "
                          f"atilan={dropped} | {summarize(resp)}", flush=True)
                else:
                    draw_overlay(frame, resp, fps, rtt_ms, dropped)

            except (requests.ConnectionError, requests.Timeout,
                    requests.HTTPError, RuntimeError) as exc:
                connected = False
                print(f"[client] BAGLANTI YOK ({type(exc).__name__}) — "
                      f"{backoff:.0f}s sonra yeniden denenecek", flush=True)
                if not args.headless:
                    cv2.putText(frame, "BAGLANTI YOK", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_WARN, 3)
                    cv2.imshow("Plaka Okuma - istemci", frame)
                    cv2.waitKey(1)
                time.sleep(backoff)                      # üstel geri çekilme
                backoff = min(backoff * 2, BACKOFF_MAX)
                last_sent = time.time()
                continue

            if not args.headless:
                cv2.imshow("Plaka Okuma - istemci", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[client] durduruldu", flush=True)
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"[client] toplam atilan kare: {dropped}", flush=True)


if __name__ == "__main__":
    main()
