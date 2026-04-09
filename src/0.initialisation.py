#!/usr/bin/env python

from rats_kinematics_utils.core.config import load_config

cfg = load_config()

print("Creating path system... \nThe following folders have been created :")

for _, path in cfg.paths : 
    if path.stem == cfg.rat_name or path.stem == cfg.bodypart:
        continue
    print("\t", path)
    path.mkdir(parents=True, exist_ok=True)