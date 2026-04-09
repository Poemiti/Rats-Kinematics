from pathlib import Path
from typing import Literal
import re, yaml
from pathlib import Path
from pydantic import Field, BaseModel, root_validator


class PathsConfig(BaseModel):
    # dynamic fields allowed
    class Config:
        extra = "allow"

    @root_validator(pre=True)
    def resolve_templates(cls, values):
        resolved = dict(values)

        # variables available in templates
        context = dict(values)

        # allow external variables later (rat_name, bodypart)
        context.update({
            "rat_name": values.get("rat_name"),
            "bodypart": values.get("bodypart"),
        })

        pattern = re.compile(r"{(.*?)}")

        def resolve(value):
            if not isinstance(value, str):
                return value

            while True:
                matches = pattern.findall(value)
                if not matches:
                    break

                for key in matches:
                    if key not in context:
                        continue
                    value = value.replace(f"{{{key}}}", str(context[key]))

            return value

        # resolve multiple passes (important for nested refs)
        for _ in range(5):
            for k, v in resolved.items():
                resolved[k] = resolve(v)
                context[k] = resolved[k]

        # convert to Path
        for k, v in resolved.items():
            resolved[k] = Path(v).expanduser().resolve()

        return resolved


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
        L1 (left lever) : LED_3
        L2 (right lever) : LED_2"""
        
        return "LED_3" if self.task == "L1" else "LED_2"



def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # inject variables into paths
    raw["paths"]["rat_name"] = raw["rat_name"]
    raw["paths"]["bodypart"] = raw["bodypart"]

    return Config(**raw)


def match_rule(meta, rules):
    value = rules.get("default")

    for rule in rules.get("rules"):
        conditions = rule.get("when", {})
    
        if all(meta.get(k) == v for k, v in conditions.items()):
            value = rule["value"]

    return value