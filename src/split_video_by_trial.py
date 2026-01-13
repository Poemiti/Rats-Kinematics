
from pathlib import Path
import pandas as pd
import time
import subprocess
import cv2

def extract_frames(input_path: Path, output_path: Path, duration: float, FPS: int = 125) -> None:
    """
    From one video, makes clips of a certain duration (in seconds) and saves frames in folders.
    """
    video = cv2.VideoCapture(str(input_path))
    frames_per_clip = int(round(FPS * duration))

    folder_count = 0
    frame_count = 0
    output_path = output_path / input_path.stem

    print("\nFrame extraction in progress...")
    print(f"Video FPS = {FPS} | Frames per clip = {frames_per_clip}")

    while True:
        success, frame = video.read()
        if not success:
            break

        clip_dir = output_path / f"clip_{folder_count:02d}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(clip_dir / f"frame_{frame_count:03d}.png"), frame)
        frame_count += 1

        if frame_count >= frames_per_clip:
            frame_count = 0
            folder_count += 1
            print(f"Frames saved in {clip_dir}!")
            break

    video.release()


def get_video_properties(video_path: Path, CLIP_DURATION: float, FPS: int = 125) -> dict:
    """
    Returns video metadata as a dictionary.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / FPS if FPS > 0 else 0

    frames_per_clip = int(FPS * CLIP_DURATION)
    n_clips = frame_count // frames_per_clip

    metadata = {
        "video_path": video_path,
        "resolution": (width, height),
        "FPS": FPS,
        "frame_count": frame_count,
        "duration_sec": duration_sec,  # seconds
        "frames_per_clip": frames_per_clip,
        "expected_clips": n_clips
    }

    cap.release()
    return metadata


def frames_to_video(frames_dir: Path, output_video: Path, FPS: float) -> None:
    """
    Converts a folder of frames to a video file.
    """
    print("\nConverting frames to video...")

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError("Cannot read first frame")

    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, FPS, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Cannot read frame {frame_path}")
        writer.write(frame)

    writer.release()
    print(f"Video saved to {output_video}, duration: {len(frame_paths)/FPS:.2f} s")


def split_video(input_path: Path, output_path: Path,
                CLIP_DURATION: float = 3,
                FPS: int = 125,
                CRF: int = 23) -> None:

    output_path.mkdir(parents=True, exist_ok=True)
    fixed_video_path = output_path / "fixed_125fps.mp4"

    print("\nSplitting video in one go ...\n")

    # ------------------ STEP 1: reinterpret frames as 125 fps + compression

    ffmpeg_args = [
        "ffmpeg",
        "-y",

        "-r", str(FPS),
        "-i", str(input_path),

        "-c:v", "libx264",
        "-crf", str(CRF),
        "-pix_fmt", "yuv420p",
        "-vsync", "cfr",

        str(fixed_video_path)
    ]
    run_ffmpeg(ffmpeg_args)

    # Re-probe FIXED video
    metadata = get_video_properties(fixed_video_path, CLIP_DURATION)
    total_duration = metadata["duration_sec"]

    print(f"\nCRF : {CRF} | Video FPS : {FPS} | Duration : {total_duration:.2f} sec")
    print(f"Number of output clips : {round(total_duration / CLIP_DURATION)}\n")

    # ------------------ STEP 2: split (NO re-encode)

    start_time = 0.0
    i = 0
    while start_time < total_duration:
        print(f"\n### CLIPPING AT {start_time:.2f} sec, clip N°{i}\n")

        ffmpeg_args = [
            "ffmpeg",
            "-y",

            "-ss", str(start_time),
            "-i", str(fixed_video_path),
            "-t", str(CLIP_DURATION),

            "-c", "copy",

            str(output_path / f"clip_{i:02d}.mp4")
        ]
        run_ffmpeg(ffmpeg_args)

        start_time += CLIP_DURATION
        i += 1

    # ------------------ STEP 3: cleanup

    fixed_video_path.unlink()




def run_ffmpeg(ffmpeg_args: list[str]) -> None : 
    try:
            subprocess.run(ffmpeg_args, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed:\n{e}")


if __name__ == "__main__":
    # ---------------------------------------------- setup path -------------------------------------------------

    OUTPUT_DATA_DIR = Path("../data")
    INPUT_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")
    DATABASE_PATH = Path("../exploration/no_KO_video_list.csv")

    # ---------------------------------------------- setup constant -------------------------------------------------

    DATABASE = pd.read_csv(DATABASE_PATH)
    VIDEO_EXEMPLE = Path(DATABASE.iloc[0]["filename"])
    CLIP_DURATION = 3  # seconds
    FPS = 125
    CRF = 28

    # ------------------------------------------- get metadata (cv2) -------------------------------------------------

    metadata = get_video_properties(VIDEO_EXEMPLE, CLIP_DURATION, FPS)
    print("\nMetadata :")
    for key, val in metadata.items() : 
        print(f"{key}       : {val}")

    # ------------------------------------ method 1 :  extract frames + video convertion -------------------------------------------------

    start_time = time.perf_counter()
    extract_frames(VIDEO_EXEMPLE, OUTPUT_DATA_DIR, CLIP_DURATION, FPS)
    end_time = time.perf_counter()

    # Convert frames back to video
    frames_dir = OUTPUT_DATA_DIR / VIDEO_EXEMPLE.stem / "clip_00"
    output_video_path = Path("./output_3sec_125FPS.mp4")

    vid_start = time.perf_counter()
    frames_to_video(frames_dir, output_video_path, FPS)
    vid_end = time.perf_counter()

    # -----------------------------------  method 2 :  directly split video  -------------------------------------------------

    output_clips_path = Path("./clips")
    split_start = time.perf_counter()
    split_video(VIDEO_EXEMPLE, output_clips_path, CLIP_DURATION, FPS, CRF)
    split_end = time.perf_counter()

    # -----------------------------------  Display perfomance -------------------------------------------------


    print(f"\nPerformance:")
    print(f"  Frame extraction time         : {(end_time - start_time):.2f} sec")
    print(f"  Video creation time           : {(vid_end - vid_start):.2f} sec")
    print(f"  Direct video splitting time   : {(split_end - split_start):.2f} sec")


    n_vid = len(DATABASE)
    n_clip_total = n_vid * metadata['expected_clips'] 
    total_time_frame_extract = ((n_clip_total * (end_time - start_time)) / 60 ) / 60 # h
    total_time_video_making = ((n_clip_total * (vid_end - vid_start)) / 60) / 60 # h
    total_time_video_splitting = ((n_vid * (split_end - split_start)) / 60) / 60 # h

    print(f"\nOVERALL TIME PERFOMANCE PREDICTION :")
    print(f"  Clip/trial duration                   : {CLIP_DURATION} sec")
    print(f"  Number of frame per clip              : {metadata['frames_per_clip']}")
    print(f"  Average expected nb of clip/video     : {metadata['expected_clips']}")
    print(f"  Number of raw video                   : {n_vid}")
    print(f"  Total number of clip                  : {n_clip_total}")
    print()
    print(f"  Total Time for frame extraction       : {total_time_frame_extract:.2f} h")
    print(f"  Total Time for video making           : {total_time_video_making:.2f} h ")
    print()
    print(f"  Total Time for direct video splitting : {total_time_video_splitting:.2f} h ")

    # Performance:
    #   Frame extraction time         : 3.28 sec
    #   Video creation time           : 1.88 sec
    #   Direct video splitting time   : 340.73 sec

    # OVERALL TIME PERFOMANCE PREDICTION :
    #   Clip/trial duration                   : 3 sec
    #   Number of frame per clip              : 375
    #   Average expected nb of clip/video     : 43
    #   Number of raw video                   : 644
    #   Total number of clip                  : 27692

    #   Total Time for frame extraction       : 25.21 h
    #   Total Time for video making           : 14.47 h 

    #   Total Time for direct video splitting : 60.95 h !!!!!!!!

    #########################################################################
    #### with compression : CRF = 25, thread = 5
    #########################################################################
    # Performance:
    #   Frame extraction time         : 3.24 sec
    #   Video creation time           : 1.89 sec
    #   Direct video splitting time   : 340.15 sec

    # OVERALL TIME PERFOMANCE PREDICTION :
    #   Clip/trial duration                   : 3 sec
    #   Number of frame per clip              : 375
    #   Average expected nb of clip/video     : 43
    #   Number of raw video                   : 644
    #   Total number of clip                  : 27692

    #   Total Time for frame extraction       : 24.93 h
    #   Total Time for video making           : 14.57 h 

    #   Total Time for direct video splitting : 60.85 h

    #########################################################################
    #### with compression : CRF = 45, no thread specified (default seems to be 12)
    #########################################################################
    # Performance:
    #   Frame extraction time         : 3.21 sec
    #   Video creation time           : 1.88 sec
    #   Direct video splitting time   : 340.64 sec

    # OVERALL TIME PERFOMANCE PREDICTION :
    #   Clip/trial duration                   : 3 sec
    #   Number of frame per clip              : 375
    #   Average expected nb of clip/video     : 43
    #   Number of raw video                   : 644
    #   Total number of clip                  : 27692

    #   Total Time for frame extraction       : 24.71 h
    #   Total Time for video making           : 14.46 h 

    #   Total Time for direct video splitting : 60.94 h 

    #########################################################################
    #### with compression : CRF = 28, 2 STEP PROCESS : 1 fixed video and then splitting
    #########################################################################

    # Performance:
    #   Frame extraction time         : 3.46 sec
    #   Video creation time           : 1.92 sec
    #   Direct video splitting time   : 37.32 sec

    # OVERALL TIME PERFOMANCE PREDICTION :
    #   Clip/trial duration                   : 3 sec
    #   Number of frame per clip              : 375
    #   Average expected nb of clip/video     : 43
    #   Number of raw video                   : 644
    #   Total number of clip                  : 27692

    #   Total Time for frame extraction       : 26.63 h
    #   Total Time for video making           : 14.75 h 

    #   Total Time for direct video splitting : 6.68 h !!!!!!
