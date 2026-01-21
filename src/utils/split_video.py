
from pathlib import Path
import pandas as pd
import time
import subprocess
import cv2

def extract_frames(input_path: Path, output_path: Path, duration: float, fps: int = None) -> None:
    """
    Extract frames every 'duration' from a single video and save them into clip folders.

    The video is read sequentially and frames are saved as PNG images.
    Frames are grouped into a clip of fixed duration (in seconds).

    If `fps` is not provided, the video frame rate is inferred
    using `get_video_properties`.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the input video file.
    output_path : pathlib.Path
        Base directory where extracted frames will be saved.
        Frames are stored under:
        ``output_path / input_path.stem / clip_XX / frame_YYY.png``.
    duration : float
        Duration of the clip to extract, in seconds.
    fps : int, optional
        Frame rate to assume for extraction. If None, the FPS
        is read from the video metadata.

    Returns
    -------
    None

    """
    if fps is None : 
        metadata = get_video_properties(input_path, duration)
        fps = metadata["fps"]

    video = cv2.VideoCapture(str(input_path))
    frames_per_clip = int(round(fps * duration))

    folder_count = 0
    frame_count = 0


    print("\nFrame extraction in progress...")
    print(f"Video FPS = {fps} | Frames per clip = {frames_per_clip}")

    while True:
        success, frame = video.read()
        if not success:
            break

        clip_dir = output_path / input_path.stem / f"clip_{folder_count:02d}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(clip_dir / f"frame_{frame_count:03d}.png"), frame)
        frame_count += 1

        if frame_count >= frames_per_clip:
            frame_count = 0
            folder_count += 1
            print(f"Frames saved in {clip_dir} !")

    video.release()


def get_video_properties(video_path: Path, CLIP_DURATION: float = 12.5, fps = None) -> dict:
    """
    Returns basic metadata (dict) and clip statistics from a video file.

    Parameters
    ----------
    video_path : pathlib.Path
        Path to the video file.
    CLIP_DURATION : float, optional
        Desired clip duration in seconds, used to estimate
        the number of clips. Default is 12.5.
    fps : int, optional
        Frame rate to use. If None, the FPS is read from the video file.

    Returns
    -------
    dict
        Dictionary containing video metadata :
        - ``video_path`` : Path
        - ``resolution`` : tuple[int, int]
        - ``fps`` : int
        - ``frame_count`` : int
        - ``duration_sec`` : float
        - ``frames_per_clip`` : int
        - ``expected_clips`` : int
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if fps is None : 
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0

    frames_per_clip = int(fps * CLIP_DURATION)
    n_clips = frame_count // frames_per_clip

    metadata = {
        "video_path": video_path,
        "resolution": (width, height),
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,  # seconds
        "frames_per_clip": frames_per_clip,
        "expected_clips": n_clips
    }

    cap.release()
    return metadata


def frames_to_video(frames_dir: Path, output_video_path: Path, fps: float = 30) -> None:
    """
    Assemble a sequence of image frames into a video file.

    Frames must be named using the pattern ``frame_XXX.png`` and
    are written in sorted order into an MP4 video.

    Parameters
    ----------
    frames_dir : pathlib.Path
        Directory containing frame images.
    output_video_path : pathlib.Path
        Directory where the output video will be written.
    fps : float, optional
        Frame rate of the output video. Default is 30.

    Returns
    -------
    None
    """

    print("\nConverting frames to video...")
    output_video_path.mkdir(exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        print(type(frame_path))
        raise RuntimeError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError("Cannot read first frame")

    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path / "clip_00.mp4"), fourcc, fps, (width, height))  ### CHANGE CLIP NAME

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise RuntimeError(f"Cannot read frame {frame_path}")
        writer.write(frame)

    writer.release()
    print(f"Video saved to {output_video_path}, duration: {len(frame_paths)/fps:.2f} s")



def split_video(input_path: Path, output_path: Path,
                CLIP_DURATION: float = 3,
                FPS: int = None,
                CRF: int = 23) -> None:
    """
    Split a video into fixed-duration clips using FFmpeg.

    The video is first re-encoded to a constant frame rate and
    compressed using H.264. It is then split into multiple clips
    without re-encoding.
    Requires FFmpeg to be installed and available in PATH.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the input video file.
    output_path : pathlib.Path
        Directory where output clips will be saved.
        Output clip names follow the pattern: ``{output_path.stem}_clip_XX.mp4``
    CLIP_DURATION : float, optional
        Duration of each clip in seconds. Default is 3.
    FPS : int, optional
        Target frame rate. If None, the original FPS is preserved.
    CRF : int, optional
        Constant Rate Factor for H.264 compression
        (lower = better quality). Default is 23.

    Returns
    -------
    None
    """

    output_path.mkdir(parents=True, exist_ok=True)
    fixed_video_path = output_path / "fixed_125fps.mp4"

    print("\nSplitting video in one go ...")

    # ------------------ STEP 1: reinterpret frames as N fps + compression
    print("\nCompressing video ...\n")
    ffmpeg_args = [
        "ffmpeg",
        "-y"]

    if FPS : 
        ffmpeg_args += ["-r", str(FPS)]

    ffmpeg_args += [
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
    if FPS is None : 
        FPS = metadata["fps"]

    print(f"\nCRF : {CRF} | Video FPS : {FPS} | Video Duration : {total_duration:.2f} sec")
    print(f"Clip duration : {CLIP_DURATION}  |  Number of output clips : {round(total_duration / CLIP_DURATION)}\n")

    # ------------------ STEP 2: split normaly (NO re-encode)

    start_time = 0.0
    i = 0
    while start_time < total_duration:
        print(f"\n# Clipping video from {start_time:.2f} - {start_time+CLIP_DURATION}, clip N°{i}\n")

        ffmpeg_args = [
            "ffmpeg",
            "-y",

            "-ss", str(start_time),
            "-i", str(fixed_video_path),
            "-t", str(CLIP_DURATION),

            "-c", "copy",

            str(output_path / f"{output_path.stem}_clip_{i:02d}.mp4")
        ]
        run_ffmpeg(ffmpeg_args)

        start_time += CLIP_DURATION
        i += 1

    # ------------------ STEP 3: cleanup

    fixed_video_path.unlink()



def run_ffmpeg(ffmpeg_args: list[str]) -> None : 
    """
    Execute an FFmpeg command.
    Requiere FFmpeg to be installed

    Parameters
    ----------
    ffmpeg_args : list of str
        Full FFmpeg command as a list of arguments.

    Returns
    -------
    None
    """

    try:
        subprocess.run(ffmpeg_args, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed:\n{e}")



if __name__ == "__main__":
    # ---------------------------------------------- setup path -------------------------------------------------

    # inputs (should exist)
    GENERATED_DATA_DIR = Path("../../exploration/data")
    DATABASE_PATH = GENERATED_DATA_DIR / "database/rat_517_H001.csv"  # if it does not exist, make one with make_database (in file_management.py)
    INPUT_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")

    # outputs
    GENERATED_FRAMES_DIR = GENERATED_DATA_DIR / "frames" 
    GENERATED_VIDEOS_DIR = GENERATED_DATA_DIR / "clips"
    GENERATED_FRAME2VIDEOS_DIR = GENERATED_DATA_DIR / "frame_to_clips"

    GENERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_FRAMES_DIR.mkdir(exist_ok=True)
    GENERATED_VIDEOS_DIR.mkdir(exist_ok=True)
    GENERATED_FRAME2VIDEOS_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------- setup constant -------------------------------------------------

    database = pd.read_csv(DATABASE_PATH)
    VIDEO_EXEMPLE = Path(database.iloc[0]["filename"])
    CLIP_DURATION = 12.5  # sec,  3 sec if fps=125, 12.5 sec if fps=30  (1 trial = 375 frames)
    FPS = 30   # used if you want to set the fps for the outputs clip
    CRF = 28   # compression value (the higher, the greater is the compression)

    # ------------------------------------------- get metadata (cv2) -------------------------------------------------

    metadata = get_video_properties(VIDEO_EXEMPLE, CLIP_DURATION)
    print("\nRaw Video Metadata :")
    for key, val in metadata.items() : 
        print(f"  {key} : {val}")


    # ------------------------------------ method 1 :  extract frames + video convertion -------------------------------------------------

    start_time = time.perf_counter()
    extract_frames(VIDEO_EXEMPLE, GENERATED_FRAMES_DIR, CLIP_DURATION)
    end_time = time.perf_counter()

    # Convert frames OF CLIP_00 ONLY back to video
    frames_dir = GENERATED_FRAMES_DIR / VIDEO_EXEMPLE.stem / "clip_00"
    output_video_path = GENERATED_FRAME2VIDEOS_DIR / VIDEO_EXEMPLE.stem 

    vid_start = time.perf_counter()
    frames_to_video(frames_dir, output_video_path, FPS)
    vid_end = time.perf_counter()

    # -----------------------------------  method 2 :  directly split video  -------------------------------------------------

    output_clips_path = GENERATED_VIDEOS_DIR / VIDEO_EXEMPLE.stem
    split_start = time.perf_counter()
    split_video(input_path=VIDEO_EXEMPLE, 
                output_path=output_clips_path, 
                CLIP_DURATION=CLIP_DURATION, 
                FPS=None ,CRF=CRF)
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


    #####################################################
    # NO FPS CHANGED : clip_00 metadata : 

    # Raw Video Metadata :
    #   video_path : /media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024/#516/02072024/Rat_#516Ambidexter_20240702_BetaMT300_LeftHemiCHR_onlyL1LeftHand_C001H001S0001/Rat_#516Ambidexter_20240702_BetaMT300_LeftHemiCHR_onlyL1LeftHand_C001H001S0001.avi
    #   resolution : (512, 512)
    #   fps : 30
    #   frame_count : 16380
    #   duration_sec : 131.04
    #   frames_per_clip : 1562
    #   expected_clips : 10

    # clip metadata :
    #   video_path : ../data/direct_clips/Rat_#516Ambidexter_20240702_BetaMT300_LeftHemiCHR_onlyL1LeftHand_C001H001S0001/clip_00.mp4
    #   resolution : (512, 512)
    #   fps : 30
    #   frame_count : 377
    #   duration_sec : 3.016
    #   frames_per_clip : 1562
    #   expected_clips : 0


    #####################################################
    # FPS CHANGED = 125 fps : clip_00 metadata : 

