import os
import shutil
from pathlib import Path

src_dirs = [
    Path("/dataset/LibriSpeech_wav/train"),
    Path("/dataset/VoxCeleb_wav/train"),
]
dest_dir = Path("/dataset/merge_libri_voxcel")
dest_dir.mkdir(parents=True, exist_ok=True)

for src in src_dirs:
    for wav_file in src.rglob("*.wav"):
        target = dest_dir / wav_file.name
        # Rename if duplicate filenames exist
        if target.exists():
            base, ext = os.path.splitext(wav_file.name)
            count = 1
            while (dest_dir / f"{base}_{count}{ext}").exists():
                count += 1
            target = dest_dir / f"{base}_{count}{ext}"
        shutil.copy2(wav_file, target)

print("✅ Merge completed successfully!")
