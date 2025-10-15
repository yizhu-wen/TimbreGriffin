import os, shutil

train_dir = "/dataset/VoxCeleb_wav/train"
val_dir = "/dataset/VoxCeleb_wav/val"
num_to_move = 4874

os.makedirs(val_dir, exist_ok=True)

# List and sort all wav files
wav_files = sorted([f for f in os.listdir(train_dir) if f.endswith(".wav")])
print("len(wav_files)", len(wav_files))

# Select last N
to_move = wav_files[-num_to_move:]

# Move files
for f in to_move:
    src = os.path.join(train_dir, f)
    dst = os.path.join(val_dir, f)
    shutil.move(src, dst)

print(f"Moved {len(to_move)} files to {val_dir}/")
