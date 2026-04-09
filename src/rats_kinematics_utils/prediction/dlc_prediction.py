import shutil
from pathlib import Path
import time
import xarray as xr
import pandas as pd



def dlc_predict_Rejane(
    model_path: Path,
    input_video_path: Path,
    temporary_path: Path,
    output_h5_path: Path,
    output_csv_path: Path,
    save_as_csv: bool = True,
) -> Path:
    """
    Run DeepLabCut analysis on a single video and return the temporary output directory.
    """

    analysis_output_path = temporary_path / f"analysis_{input_video_path.stem}"
    analysis_output_path.mkdir(parents=True, exist_ok=True)

    deeplabcut.analyze_videos(
        f"{model_path}/config.yaml",
        [str(input_video_path)],
        save_as_csv=save_as_csv,
        destfolder=str(analysis_output_path),
    )

    move_outputs(analysis_output_path, output_h5_path, output_csv_path)   
    cleanup_temp_directory(analysis_output_path)




def move_outputs(
    analysis_output_path: Path,
    output_h5_path: Path | None = None,
    output_csv_path: Path | None = None,
):
    """
    Move DeepLabCut output files from the temporary directory to final destinations.
    """
    if not output_h5_path and not output_csv_path:
        raise ValueError(
            "At least one of output_h5_path or output_csv_path must be provided."
        )

    h5_file = next(analysis_output_path.glob("*.h5"), None)
    csv_file = next(analysis_output_path.glob("*.csv"), None)

    if output_h5_path and h5_file:
        shutil.move(str(h5_file), str(output_h5_path))

    if output_csv_path and csv_file:
        shutil.move(str(csv_file), str(output_csv_path))



def cleanup_temp_directory(analysis_output_path: Path):
    """
    Remove the temporary analysis directory.
    """
    if analysis_output_path.exists():
        shutil.rmtree(analysis_output_path)



def dlc_predict_Julien(model_path: Path, 
                       video_path: Path, 
                       output_csv_path : Path = None) -> xr.DataArray:
    import tempfile
    import deeplabcut
    from deeplabcut.pose_estimation_pytorch import set_load_weights_only

    with tempfile.TemporaryDirectory() as dlc_dest:
        # print(dlc_dest)
        deeplabcut.analyze_videos(
            f'{model_path}/config.yaml',
            [str(video_path)],
            save_as_csv=False,
            # gputouse=0,
            destfolder=dlc_dest
        )

        h5_file = next(Path(dlc_dest).glob("*.h5"), None)
        df = pd.read_hdf(h5_file)

    df.index.name="frame_num"

    if output_csv_path : 
        print(df)
        df.to_csv(output_csv_path)

    res =  df.stack("scorer").stack("bodyparts").stack("coords").to_xarray()

    if res.sizes["scorer"] !=1:
        raise Exception(f"Multiple scorers not supported, got {res.sizes['scorer']}")
    res = res.isel(scorer=0, drop=True)

    return res






if __name__ == "__main__":

    print("No main")


