"""Verify a complete ZIP member while the official corpus keeps downloading.

Used ONLY for a two-update native adapter smoke. The formal alignment still
requires the complete official ten-subject archive and its manifest.
"""
from pathlib import Path
import hashlib
import json
import struct
import time
import zlib

p=Path(__file__).resolve().parent
archive=p/'assets_20261113/mhealth/mhealth.part'
output=p/'assets_20261113/mhealth_partial_verified'
output.mkdir(exist_ok=True)
while not (p/'assets_20261113/mhealth/ASSETS.json').exists():
    if not archive.exists():time.sleep(20);continue
    with archive.open('rb') as source:
        header=source.read(30)
        while len(header)==30:
            sig,version,flags,method,tm,date,crc,compressed,size,nlen,xlen=struct.unpack('<4s5H3I2H',header)
            if sig!=b'PK\x03\x04' or flags&9:break
            name=source.read(nlen).decode('utf-8');source.read(xlen)
            start=source.tell()
            if archive.stat().st_size<start+compressed:break
            if name.endswith('.log'):
                raw=source.read(compressed)
                data=zlib.decompress(raw,-15) if method==8 else raw
                assert len(data)==size and zlib.crc32(data)&0xffffffff==crc
                target=output/Path(name).name;target.write_bytes(data)
                manifest=dict(status='partial_member_verified_for_smoke_only',source='https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip',
                    member=name,compressed_offset=start,compressed_bytes=compressed,uncompressed_bytes=size,crc32=f'{crc:08x}',
                    sha256=hashlib.sha256(data).hexdigest(),formal_alignment_allowed=False)
                (output/'PARTIAL_ASSETS.json').write_text(json.dumps(manifest,indent=2)+'\n')
                print('Verified complete official MHealth member for native smoke only',manifest,flush=True)
                raise SystemExit(0)
            source.seek(start+compressed);header=source.read(30)
    time.sleep(20)
