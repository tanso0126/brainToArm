"""Path B ONLY — run on WINDOWS if the PolyG-I refuses to be a plain COM port.

If eeg_detect.py shows no streaming virtual COM port, the device is reached
through LAXTHA's Windows driver DLL (LXSMWD12.dll). This server runs on a Windows
box (or a Windows VM with the USB device passed through), pulls the RAW LXSDF
byte stream from the DLL, and forwards those bytes unchanged over TCP to the Mac
running orchestrator.py (EEG_SOURCE='tcp'). The Mac side runs the SAME LXSDFParser,
so nothing about the decoding differs between Path A and Path B.

NO ROUTER: run both ends on one machine via 127.0.0.1, or connect the two
machines with a single ethernet cable and use their link-local IPs.

The DLL call names below follow LAXTHA's documented LXSMWD12 API pattern. If a
signature differs, the developer manual (laxtha.net, "LXSMWD12 Developer Manual")
has the exact prototype — change only the ctypes lines flagged CONFIRM.
"""
import socket
import ctypes
import time
import argparse

HOST, PORT = "127.0.0.1", 9000
DLL_NAME = "LXSMWD12.dll"
READ_CHUNK = 512          # bytes to pull per DLL read


class LaxthaDLL:
    """Thin wrapper over LXSMWD12.dll returning raw LXSDF bytes."""

    def __init__(self, dll_name=DLL_NAME):
        self.dll = ctypes.WinDLL(dll_name)     # DLL must be on PATH / cwd
        # CONFIRM signatures against the LXSMWD12 Developer Manual:
        #   int  LXSMWD12_Open(int port)      -> handle/status
        #   int  LXSMWD12_Start(void)
        #   int  LXSMWD12_GetData(byte* buf, int maxlen) -> bytes written
        #   void LXSMWD12_Stop(void); LXSMWD12_Close(void)
        self._buf = (ctypes.c_ubyte * 4096)()
        get = getattr(self.dll, "LXSMWD12_GetData", None)
        if get is not None:
            get.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
            get.restype = ctypes.c_int

    def open(self, port=0):
        if hasattr(self.dll, "LXSMWD12_Open"):
            self.dll.LXSMWD12_Open(port)
        if hasattr(self.dll, "LXSMWD12_Start"):
            self.dll.LXSMWD12_Start()

    def read(self, n=READ_CHUNK):
        if not 0 < n <= len(self._buf):
            raise ValueError(f"read size must be in [1, {len(self._buf)}]")
        get = getattr(self.dll, "LXSMWD12_GetData", None)
        if get is None:
            raise RuntimeError("LXSMWD12_GetData not found; check manual for the "
                               "correct read function name and fix this line")
        got = get(self._buf, n)
        if got <= 0:
            return b""
        if got > n:
            raise RuntimeError(f"DLL reported {got} bytes for a {n}-byte buffer request")
        return bytes(self._buf[:got])

    def close(self):
        if hasattr(self.dll, "LXSMWD12_Stop"):
            self.dll.LXSMWD12_Stop()
        if hasattr(self.dll, "LXSMWD12_Close"):
            self.dll.LXSMWD12_Close()


def serve(host=HOST, port=PORT, dll_name=DLL_NAME):
    dev = LaxthaDLL(dll_name)
    dev.open()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, port))
            srv.listen(1)
            print(f"[win-eeg] listening {host}:{port}")
            print("[win-eeg] device started, waiting for orchestrator...")
            while True:
                conn, addr = srv.accept()
                print(f"[win-eeg] client {addr}")
                with conn:
                    try:
                        while True:
                            data = dev.read()
                            if data:
                                conn.sendall(data)
                            else:
                                time.sleep(0.002)
                    except (ConnectionResetError, BrokenPipeError):
                        print("[win-eeg] client gone; waiting for reconnect")
    finally:
        dev.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST,
                    help="listen address; use 0.0.0.0 only for a direct remote Mac")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--dll", default=DLL_NAME)
    args = ap.parse_args()
    if not 1 <= args.port <= 65535:
        ap.error("--port must be in [1, 65535]")
    try:
        serve(args.host, args.port, args.dll)
    except KeyboardInterrupt:
        print("\n[win-eeg] stopped")


if __name__ == "__main__":
    main()
