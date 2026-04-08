#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib, time, sys
import plotly.graph_objects as go
from collections import Counter

from rats_kinematics_utils.config import load_config
from rats_kinematics_utils.pipeline_maker import load_metrics, print_analysis_info



# ------------------------------------ setup ---------------------------------------

cfg = load_config()
print_analysis_info(cfg, "Preprocessing")

RAT_NAME = cfg.rat_name

output_dir = cfg.paths.metrics / RAT_NAME
output_dir.mkdir(parents=True, exist_ok=True)


yaml_filenames = list((cfg.paths.metrics / RAT_NAME).rglob("*.yaml"))
print(f"Number of metadata files: {len(yaml_filenames)}")

joblib_filenames = list((cfg.paths.metrics / RAT_NAME).glob("*.joblib"))
ntrials = 0

for file in joblib_filenames : 
    file = Path(file)

    meta = joblib.load(file)
    ntrials += len(meta)
print(f"Number of joblib metadata files: {ntrials}")


data = []
noCue_video = {}

for f in yaml_filenames:
    with open(f, "r") as file:
        meta = yaml.safe_load(file)
        data.append(meta)

        if meta["cue_type"] == "NoCue" : 
            noCue_video[f] = meta["filename_clips"]


transitions = Counter()

for d in data:
    c = d["condition"]
    ls = d["laser_state"] 
    li = "0mW" if d["laser_intensity"] == "NOstim" else  d["laser_intensity"] + "_" +  ls
    ct = d["cue_type"]
    
    transitions[(c, ls)] += 1
    transitions[(ls, li)] += 1
    transitions[(li, ct)] += 1


nodes = list(set([x for pair in transitions for x in pair]))
node_indices = {node: i for i, node in enumerate(nodes)}

sources = []
targets = []
values = []

for (src, target), count in transitions.items():
    sources.append(node_indices[src])
    targets.append(node_indices[target])
    values.append(count)


import random

# assign a color per node
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# choose a colormap
cmap = cm.get_cmap("inferno")

# normalize over number of nodes
norm = mcolors.Normalize(vmin=0, vmax=len(nodes)-1)

# assign a color to each node
node_colors = {
    node: cmap(norm(i))  # RGBA tuple (0-1)
    for i, node in enumerate(nodes)
}

# convert RGBA (0-1) → Plotly rgba string
def to_rgba_str(rgba, alpha=0.8):
    r, g, b, _ = rgba
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha})"

node_color_list = [to_rgba_str(node_colors[n]) for n in nodes]

# links take color from source node
link_colors = [
    to_rgba_str(node_colors[nodes[s]], alpha=0.4)  # more transparent
    for s in sources
]

fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        label=nodes, 
        color=node_color_list
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors,
        customdata=values,  # store counts
        hovertemplate="From %{source.label} → %{target.label}<br>Count: %{customdata}<extra></extra>"
    )
))

fig.show()


for k, v in noCue_video.items():
    print() 
    print(k)
    print(v)

print(f"\nNumber of NoCue videos : {len(noCue_video)}")