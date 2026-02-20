from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, validator
import yaml


class PathsConfig(BaseModel):
    model: Path
    raw_videos: Path
    metrics: Path
    figures: Path
    data: Path
    clips: Path
    coords: Path
    database: Path
    luminosity: Path
    report: Path
    frames: Path
    h5: Path

    # @field_validator("*", mode="before")
    @validator("*")
    @classmethod
    def expand_paths(cls, v):
        return Path(v).expanduser().resolve()



class Config(BaseModel):
    rat_name: str
    bodypart: str
    view: Literal["left", "right"]
    task: Literal["L1", "L2"]

    threshold: float = Field(..., ge=0.0, le=1.0)
    fps: int = Field(..., gt=0)
    clip_length: float = Field(..., gt=0)
    laser_on_duration: float = Field(..., gt=0)
    max_lost_coords: int = Field(..., ge=0)
    frame_width_px: int = Field(..., gt=0)

    paths: PathsConfig

    @property
    def frame_width_cm(self) -> float:
        return 8.7 if self.view == "left" else 8.3

    @property
    def cm_per_pixel(self) -> float:
        return self.frame_width_cm / self.frame_width_px
    
    @property
    def task_pad(self) -> str:
        """return which pad we should be looking at depending on the task
        L1 (left lever) : PAD_3
        L2 (right lever) : PAD_2"""
        
        return "PAD_3" if self.task == "L1" else "PAD_2"


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        return Config(**yaml.safe_load(f))
