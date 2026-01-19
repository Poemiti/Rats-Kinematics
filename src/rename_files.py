from pathlib import Path

PREDICTION_DIR = Path("../data/clips")

for video_dir in PREDICTION_DIR.iterdir():
    if not video_dir.is_dir():
        continue

    print(video_dir.stem)

    for clip_path in video_dir.iterdir():
        if not clip_path.is_file():
            continue

        new_name = clip_path.with_name(
            f"{video_dir.stem}_{clip_path.stem}{clip_path.suffix}")

        print(new_name)

        if new_name.exists():
            raise FileExistsError(new_name)

        clip_path.rename(new_name)

        
        
        
