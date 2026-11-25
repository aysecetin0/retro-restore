"""RetroRestore web uygulamasını başlatır.

    python run.py                 # http://127.0.0.1:8000
    python run.py --host 0.0.0.0 --port 8080
"""
import argparse

from retrorestore.config import settings
from retrorestore.core.device import get_device_info
from retrorestore.web import create_app


def main():
    ap = argparse.ArgumentParser(description="RetroRestore web sunucusu")
    ap.add_argument("--host", default=settings.host)
    ap.add_argument("--port", type=int, default=settings.port)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    info = get_device_info()
    print("=" * 60)
    print(" RetroRestore — Eski Fotoğraf ve Video İyileştirme")
    print(f" Cihaz : {info.device.upper()} ({info.device_name})  FP16={info.fp16}  tile={info.default_tile}")
    if not info.torch_available:
        print(" UYARI : PyTorch yok → Real-ESRGAN yerine klasik yöntem kullanılacak.")
    print(f" Adres : http://{a.host}:{a.port}")
    print("=" * 60)
    # threaded=True: API istekleri, işçi iş parçacığı çalışırken de yanıtlanır
    create_app().run(host=a.host, port=a.port, debug=a.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
