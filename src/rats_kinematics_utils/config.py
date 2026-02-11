from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator
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
    frames: Path
    h5: Path

    @field_validator("*", mode="before")
    @classmethod
    def expand_paths(cls, v):
        return Path(v).expanduser().resolve()



class Config(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)
    bodypart: str
    fps: int = Field(..., gt=0)
    clip_length: float = Field(..., gt=0)
    laser_on_duration: float = Field(..., gt=0)
    max_lost_coords: int = Field(..., ge=0)
    view: Literal["left", "right"]
    frame_width_px: int = Field(..., gt=0)

    paths: PathsConfig

    @property
    def frame_width_cm(self) -> float:
        return 8.7 if self.view == "left" else 8.3

    @property
    def cm_per_pixel(self) -> float:
        return self.frame_width_cm / self.frame_width_px


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        return Config(**yaml.safe_load(f))
