"""LXSDF T2A packet parser for LAXTHA biosignal devices (PolyG-I).

Implemented from the official spec: http://laxtha.net/packet-lxsdf-t2a/
and github.com/LAXTHA/LXSDF (LXD12_LXSDFT2 / LXD10_LXSDFT2A standards).

Packet byte layout (stream mode), each index = 1 byte, sent in order:
    0   0xFF   SyncByte0
    1   0xFE   SyncByte1
    2   PPD    Packet Property Data (0..15 => stream mode)
    3   PUD0   Packet Unit Data 0
    4   PC     Packet Count (+1 per packet -> use to detect drops)
    5   PUD1   Packet Unit Data 1
    6   PCD    Packet Cyclic Data
    7   flags  CRD(bit6) / PUD2(bits5-3) / PCDT(bits2-0)
    8   PSD1   channel 0 High byte
    9   PSD0   channel 0 Low  byte
    10  PSD1   channel 1 High byte
    11  PSD0   channel 1 Low  byte
    ...        2 bytes per channel, High then Low, for all channels
    N          last channel Low byte

Sync bytes 0xFF 0xFE only ever appear as a pair at a packet start (high bytes
are constrained to <=253 by the standard), so we resync by scanning for FF FE.

Feed raw bytes from ANY transport (USB serial = Path A, TCP from the Windows
DLL bridge = Path B). Same parser both ways. Yields per-packet channel samples.
"""

HEADER_BYTES = 8          # indices 0..7 before stream data begins
SYNC = b"\xFF\xFE"


class LXSDFParser:
    """Streaming, resynchronizing parser. Push bytes with feed(); it yields
    (channels_list, packet_count) tuples as complete packets arrive."""

    def __init__(self, total_channels=None):
        # total_channels: number of 2-byte channel slots per packet. If None,
        # auto-detect by measuring the gap between two consecutive sync headers.
        self.total_channels = total_channels
        self.buf = bytearray()
        self._last_pc = None
        self.dropped = 0

    def _packet_len(self):
        return HEADER_BYTES + 2 * self.total_channels

    def _autodetect(self):
        """Find two consecutive FF FE and infer channel count from their gap."""
        first = self.buf.find(SYNC)
        if first < 0:
            # keep only a tail that might contain a partial sync
            if len(self.buf) > 4096:
                del self.buf[:-1]
            return False
        second = self.buf.find(SYNC, first + 2)
        if second < 0:
            return False
        gap = second - first
        if gap < HEADER_BYTES + 2 or (gap - HEADER_BYTES) % 2 != 0:
            # false sync inside data; drop the first byte and retry later
            del self.buf[:first + 1]
            return False
        self.total_channels = (gap - HEADER_BYTES) // 2
        del self.buf[:first]      # align buffer to a packet boundary
        return True

    def feed(self, data):
        self.buf.extend(data)
        out = []
        if self.total_channels is None:
            if not self._autodetect():
                return out

        plen = self._packet_len()
        while True:
            i = self.buf.find(SYNC)
            if i < 0:
                # no sync in buffer; keep a small tail
                if len(self.buf) > plen * 4:
                    del self.buf[:-1]
                break
            if i > 0:
                del self.buf[:i]          # discard bytes before sync
            if len(self.buf) < plen:
                break                     # wait for a full packet
            pkt = bytes(self.buf[:plen])
            # verify the NEXT packet also starts with sync where expected; if not,
            # this was a false sync — skip one byte and rescan.
            if len(self.buf) >= plen + 2 and self.buf[plen:plen + 2] != SYNC:
                del self.buf[:1]
                continue
            del self.buf[:plen]
            out.append(self._decode(pkt))
        return out

    def _decode(self, pkt):
        pc = pkt[4]
        if self._last_pc is not None:
            expected = (self._last_pc + 1) & 0xFF
            if pc != expected:
                self.dropped += (pc - expected) & 0xFF
        self._last_pc = pc
        chans = []
        off = HEADER_BYTES
        for _ in range(self.total_channels):
            hi = pkt[off]
            lo = pkt[off + 1]
            chans.append((hi << 8) | lo)
            off += 2
        return chans, pc


def build_packet(channels, pc=0, ppd=0):
    """Encode a stream-mode LXSDF T2A packet. Used by tests and the mock source
    so the exact same parser path is exercised without hardware."""
    pkt = bytearray([0xFF, 0xFE, ppd & 0x0F, 0x00, pc & 0xFF, 0x00, 0x00, 0x00])
    for v in channels:
        v &= 0xFFFF
        hi = (v >> 8) & 0xFF
        lo = v & 0xFF
        # spec keeps high bytes <=253 so they can't fake a sync pair
        if hi >= 0xFE:
            hi = 0xFD
        pkt.append(hi)
        pkt.append(lo)
    return bytes(pkt)
