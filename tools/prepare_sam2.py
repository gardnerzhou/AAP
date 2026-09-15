from pathlib import Path
import base64
import io
import lzma
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PART_DIR = ROOT / "vendor" / "parts"
PARTS = [PART_DIR / f"sam2_{i:02d}.b64" for i in range(9)]

missing = [str(p.relative_to(ROOT)) for p in PARTS if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing bundled SAM2 source parts: {missing}")

encoded = "".join(p.read_text(encoding="utf-8").strip() for p in PARTS)
archive = lzma.decompress(base64.b64decode(encoded))

with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tf:
    root_resolved = ROOT.resolve()
    for member in tf.getmembers():
        target = (ROOT / member.name).resolve()
        if root_resolved not in target.parents and target != root_resolved:
            raise RuntimeError(f"Unsafe archive path: {member.name}")
    tf.extractall(ROOT)

print(f"Prepared SAM2 source at: {ROOT / 'sam2_train'}")
