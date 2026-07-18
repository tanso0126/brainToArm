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

HOST, PORT = "0.0.0.0", 9000
DLL_NAME = "LXSMWD12.dll"
READ_CHUNK = 512          # bytes to pull per DLL read


class LaxthaDLL:
    """Thin wrapper over LXSMWD12.dll returning raw LXSDF bytes."""

    def __init__(self):
        self.dll = ctypes.WinDLL(DLL_NAME)     # DLL must be on PATH / cwd
        # CONFIRM signatures against the LXSMWD12 Developer Manual:
        #   int  LXSMWD12_Open(int port)      -> handle/status
        #   int  LXSMWD12_Start(void)
        #   int  LXSMWD12_GetData(byte* buf, int maxlen) -> bytes written
        #   void LXSMWD12_Stop(void); LXSMWD12_Close(void)
        self._buf = (ctypes.c_ubyte * 4096)()

    def open(self, port=0):
        if hasattr(self.dll, "LXSMWD12_Open"):
            self.dll.LXSMWD12_Open(port)
        if hasattr(self.dll, "LXSMWD12_Start"):
            self.dll.LXSMWD12_Start()

    def read(self, n=READ_CHUNK):
        get = getattr(self.dll, "LXSMWD12_GetData", None)
        if get is None:
            raise RuntimeError("LXSMWD12_GetData not found; check manual for the "
                               "correct read function name and fix this line")
        got = get(self._buf, n)
        if got <= 0:
            return b""
        return bytes(self._buf[:got])

    def close(self):
        if hasattr(self.dll, "LXSMWD12_Stop"):
            self.dll.LXSMWD12_Stop()
        if hasattr(self.dll, "LXSMWD12_Close"):
            self.dll.LXSMWD12_Close()


def serve():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT)); srv.listen(1)
    print(f"[win-eeg] listening {HOST}:{PORT}")
    dev = LaxthaDLL()
    dev.open()
    print("[win-eeg] device started, waiting for orchestrator...")
    try:
        while True:
            conn, addr = srv.accept()
            print(f"[win-eeg] client {addr}")
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


if __name__ == "__main__":
    serve()
