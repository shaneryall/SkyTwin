"""Print the SHA-256 of a FITS file's data, without its header, and the file's name.

The header holds the run's own times, so only the pixels can be compared between platforms.
Header cards are 80 bytes in 2880-byte blocks; the data starts at the block after END.
"""

import hashlib
import sys
from pathlib import Path

raw = Path(sys.argv[1]).read_bytes()
end = next(i for i in range(0, len(raw), 80) if raw[i : i + 8] == b"END     ")
start = (end // 2880 + 1) * 2880
print(hashlib.sha256(raw[start:]).hexdigest(), Path(sys.argv[1]).stem)
