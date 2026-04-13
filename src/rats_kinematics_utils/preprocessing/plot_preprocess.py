#!/usr/bin/env python

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import joblib, yaml
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from tqdm import tqdm
import seaborn as sns

# ==================================== display hyperparameter ===========================================


custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme("paper", style="ticks", rc=custom_params, palette="pastel")

LASER_COLOR = "lightpink"
LINE_COLOR = "gray"
AVG_LINE_COLOR = "navy"


# ==================================== functions ===========================================

def _outlier_proportion(df):
    props = {}

    for step, values in df.groupby("step", sort=False)["n"]:
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((values < lower) | (values > upper)).sum()
        props[step] = f"{outliers}/{len(values)}"
    return props



def _plot_traj(coord, offset, label, color, ax: plt.axes = None, marker: str = None):

    x = coord["x"] - offset
    y = coord["y"] - offset

    if ax is not None : 
        ax.plot(x, y, label=label, color=color)
        if marker : 
            ax.scatter(x, y,marker=marker)
    else : 
        plt.plot(x, y, label=label, color=color)
        if marker : 
            ax.scatter(x, y,marker=marker)



def _plot_xy(axes, coords, offset, color,  marker, label, time_pad_off= None) -> None :
    
    t = coords["t"]
    x = coords["x"] + offset
    y = coords["y"] - offset

    axes[0].plot(t, x, marker=marker, color=color, label=label)
    axes[1].plot(t, y, marker=marker, color=color)

    if time_pad_off : 
        axes[0].axvline(time_pad_off, color='k', lw=0.8, ls='--', label="time pad off")
        axes[1].axvline(time_pad_off, color='k', lw=0.8, ls='--')


def _plot_outliers(axes, coords, params) -> None : 

    t = coords["t"]
    x = coords["x"] 
    y = coords["y"] 

    dists, threshold, mask = params

    axes[0].plot(t, x, marker='|', color="#74a9cf", alpha=0.5 )
    axes[0].scatter(t[mask], x[mask], marker="x", color='red')
    axes[0].set_ylabel("x")
    axes[0].set_title("Outlier detection based on euclidian distance")

    # y(t)
    axes[1].plot(t, y, marker='|', color="#74a9cf", label="raw points", alpha=0.5)
    axes[1].scatter(t[mask], y[mask], marker="x", color='red', label="outliers")
    axes[1].set_ylabel("y")
    axes[1].invert_yaxis()
    axes[1].legend()

    # distances
    axes[2].plot(t, dists, marker='o', color="#0570b0", label="distance")
    axes[2].axhline(threshold, color="#fb6a4a", linestyle="--", linewidth=1 ,label="threshold")
    axes[2].set_ylabel("Euclidean distance (pixel)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[2].set_xticks([0, 0.5])
    axes[2].set_yticks([threshold])



def make_interpolation_figures(interpolated_coords, 
                               likelihood_filtered_coords,
                               outlier_filtered_coords,
                               raw_coords,
                               time_pad_off,
                               title, 
                               save_as):
    fig = plt.figure(figsize=(10,6))
    gs = fig.add_gridspec(2, 2)

    offset = 10 # pixel

    ax_xt = fig.add_subplot(gs[0,0])      # x(t)
    ax_yt = fig.add_subplot(gs[1,0])      # y(t)
    ax_traj = fig.add_subplot(gs[:,1])    # trajectory spans both rows
    # ax_speed = fig.add_subplot(gs[2, 0])

    _plot_xy([ax_xt, ax_yt], interpolated_coords, 3*offset,"#0570b0", "|", "3.interpolate")
    _plot_xy([ax_xt, ax_yt], likelihood_filtered_coords, 2*offset,"#74a9cf", "|", "2.likelihood")
    _plot_xy([ax_xt, ax_yt], outlier_filtered_coords, 1*offset, "#bdc9e1","|", "1.outlier")
    _plot_xy([ax_xt, ax_yt], raw_coords, 0*offset, "#d1cbdc","|", "0.raw", time_pad_off)
    
    pad_off_frame = int((time_pad_off - 0.1)* 125)
    off_frame = int((time_pad_off + 0.4) * 125)

    _plot_traj(raw_coords[pad_off_frame : off_frame], 0*offset, "raw", "#d1cbdc", ax_traj)
    _plot_traj(interpolated_coords[pad_off_frame : off_frame], 0*offset, "interpolate", "#0570b0" ,ax_traj, "|")

    # ax_speed.plot(raw_coords["t"], speed, label="speed", marker="|")
    # ax_speed.plot(raw_coords["t"][peaks], speed[peaks], "x")

    ax_xt.set_ylabel("x")
    ax_xt.legend()
    ax_xt.set_xlim(time_pad_off - 0.1, time_pad_off + 0.4)
    ax_xt.set_title(title)

    ax_yt.set_ylabel("y")
    ax_yt.set_xlim(time_pad_off - 0.1, time_pad_off + 0.4)
    ax_yt.invert_yaxis()
    ax_yt.set_xlabel("time (s)")

    ax_traj.set_title("Trajectory comparaison between\nraw and final interpolated")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.invert_yaxis()
    ax_traj.legend()
    ax_traj.set_aspect("equal")
    
    # ax_speed.set_ylabel("time")
    # ax_speed.set_xlabel("time (s)")
    # ax_speed.set_xlim(0, 0.5)
    # ax_speed.legend()

    for ax in [ax_xt, ax_yt, ax_traj]:
        ax.set_xticks([])
        ax.set_yticks([])

    ax_yt.set_xticks([time_pad_off - 0.1, time_pad_off + 0.4])

    plt.tight_layout()
    fig.savefig(save_as)
    plt.close()




def make_outlier_figures(raw_coords, params, save_as): 
    fig, axes = plt.subplots(3,1, figsize=(8,6), sharex=True)
    _plot_outliers(axes, raw_coords, params)
    plt.tight_layout()
    plt.gca().set_xlim(0,0.5)
    fig.savefig(save_as)
    plt.close()



def metadata_report(clips_folder, metadata_folder: str, show_noCue: bool = False) : 
    from collections import Counter

    yaml_filenames = list(clips_folder.rglob("*.yaml"))
    print(f"Number of metadata files: {len(yaml_filenames)}")

    joblib_filenames = list(metadata_folder.glob("*.joblib"))
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
        v = d["camera_view"] + " view"
        t = d["rat_type"]
        c = d["condition"]
        ls = d["laser_state"] 
        li = "0mW" if d["laser_intensity"] == "NOstim" else  d["laser_intensity"] + "_" +  ls
        ct = d["cue_type"]
        
        transitions[(v, t)] += 1
        transitions[(t, c)] += 1
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


    cmap = cm.get_cmap("inferno")
    norm = mcolors.Normalize(vmin=0, vmax=len(nodes)-1)

    node_colors = {
        node: cmap(norm(i))  # RGBA tuple (0-1)
        for i, node in enumerate(nodes)
    }

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

    if show_noCue: 
        for k, v in noCue_video.items() : 
            print(f"\n{k}\n{v}")

    print(f"\nNumber of NoCue videos : {len(noCue_video)}")

    return fig






from rats_kinematics_utils.core.file_utils import make_output_path
from datetime import datetime
import pandas as pd
from rats_kinematics_utils.preprocessing.preprocess import open_DLC_results, filter_likelihood, filter_outliers, interpolate_data


def plot_likelihood_across_frames(cfg, filenames) : 
    
    for i, session in enumerate(filenames):
        session_data = joblib.load(session)
        folder = session.stem

        print(f"Processing {folder}")

        for trial in tqdm(session_data, desc="Trials") :
            # date = datetime.fromisoformat(trial["date"])
            # if date.month == 5 : 
            #     continue

            # loading
            coords_path = Path(trial['filename_coords'])
            raw_coords = open_DLC_results(coords_path)
            bodyparts = raw_coords.columns.get_level_values(0).unique()

            raw_coords = raw_coords[cfg.bodypart]
            _, likelihood_threshold = filter_likelihood(raw_coords, cfg.threshold, percentile=0.80)

            data = pd.DataFrame(raw_coords)

            fig, ax = plt.subplots()

            ax.plot(data["likelihood"], color="lightblue")

            ax.axhline(likelihood_threshold, linestyle="--", color="red", 
                    label="likelihood threshold", lw="0.8")

            ax.set_title(f"likelihood distribution across frame\nclip number: {trial['nb_clip']}, likelihood threshold: {likelihood_threshold:.2f}")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Likelihood")
            ax.legend(loc="lower right")

            plt.xticks()
            plt.tight_layout()
            fig.savefig(make_output_path(cfg.paths.analysis / folder / "likelihood_distri", f"{trial['name']}_likelihood.png"))
            plt.close()



def plot_likelihood_distribution(cfg, yaml_filenames) : 

    likelihood_distri = []

    for f in tqdm(yaml_filenames, desc="collecting data"):
        with open(f, "r") as file:
            trial = yaml.safe_load(file)

            # date = datetime.fromisoformat(trial["date"])
            # if date.month == 5 : 
            #     continue

            # loading
            coords_path = Path(trial['filename_coords'])
            raw_coords = open_DLC_results(coords_path)
            bodyparts = raw_coords.columns.get_level_values(0).unique()

            for bp in bodyparts[1:]:
                likelihoods = raw_coords[bp]["likelihood"]

                for val in likelihoods:
                    likelihood_distri.append({
                        "bodypart": bp,
                        "likelihood": val
                    })

    data = pd.DataFrame(likelihood_distri)


    fig, ax = plt.subplots()

    sns.violinplot(
        data=data,
        x="bodypart",
        y="likelihood",
        inner="quart",
        ax=ax
    )

    ax.axhline(cfg.threshold, linestyle="--", color="red", label="likelihood threshold", lw="0.8")

    ax.set_title(f"Distribution of likelihood across bodyparts of rat {cfg.rat_name}\nlikelihood threshold={cfg.threshold:.2f}")
    ax.set_xlabel("Bodyparts")
    ax.set_ylabel("Likelihood")
    ax.legend(loc="lower right")

    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig(make_output_path(cfg.paths.rat_root, f"{cfg.rat_name}_bodypart_likelihood_distribution.png"))
    plt.show()
    plt.close()



def plot_trial_success_distri(cfg, filenames) : 

    all_state = {"raw" : 0,
                "interpolate": 0,
                "rejected": 0, 
                "None": 0}

    for session in filenames :
        
        for trial in joblib.load(session) :
            body = trial.get(cfg.bodypart)

            if body is None : 
                print("BODY NONE")
                continue

            success = body.get("trial_success")
            state = body.get("xy_state")

            if state is None : 
                print(trial["name"])
                all_state["None"] += 1
                continue

            all_state[state] += 1

    df = pd.DataFrame({
        "State": list(all_state.keys()),
        "Count": list(all_state.values())
    })

    order = ["rejected", "raw", "interpolate", "None"]

    fig = plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        data=df,
        x="State",
        y="Count",
        palette="pastel",
        order=order,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%d')

    ax.set_title(f"Trial State Distribution of Rat {cfg.rat_name}")
    ax.set_xlabel("State")
    ax.set_ylabel("Number of Trials")

    plt.tight_layout()
    fig.savefig(make_output_path(cfg.paths.analysis, f"{cfg.rat_name}_trial_state_distribution.png"))
    plt.show()
    plt.close()






def plot_preprocess_lost_points(cfg, filenames) : 
    import numpy as np

    folder_names = {}

    for i, session in enumerate(filenames) : 
        folder_name = session.stem
        counts = []

        session_data = joblib.load(session)
        print(f"\nProcessing {folder_name}")
        for trial in tqdm(session_data, desc="Trials"):

            # date = datetime.fromisoformat(trial["date"])
            # if date.month == 5 : 
            #     continue
                
            # get all the path (coordinates, luminosity and video clips)
            coords_path = trial["filename_coords"]
            trial_name = trial["name"]

            # get coords + filtration
            raw_coords = open_DLC_results(coords_path)
            raw_coords = raw_coords[cfg.bodypart].copy()
            raw_coords = raw_coords.assign(t=np.arange(len(raw_coords)) / cfg.fps)
            
            outlier_filtered_coords, params = filter_outliers(raw_coords, stat_method='eucli')
            likelihood_filtered_coords, likelihood_threshold = filter_likelihood(outlier_filtered_coords, cfg.threshold)
            interpolated_coords = interpolate_data(likelihood_filtered_coords, method="spline", max_gap=5)

            # save number of removal
            counts.append({"id": trial_name, "step": "raw", "n": 0})
            steps = {
                "outlier": outlier_filtered_coords,
                "likelihood": likelihood_filtered_coords,
                "interpolate": interpolated_coords,
            }

            for step, df in steps.items():
                counts.append({
                    "id": trial_name,
                    "step": step,
                    "n": len(raw_coords) - df['x'].count().sum(),
                })

        # final saving after the end of the loop
        if folder_name in folder_names.keys() : 
            folder_names[folder_name].extend(counts)
        else : 
            folder_names[folder_name] = counts


    ########################## distribution ######################

    all_counts = []

    for folder, count in folder_names.items() : 

        if len(count) == 0: 
            print(f"{folder}: no data")
            print(count)
            continue

        all_counts.extend(count)
        data = pd.DataFrame(count)

        fig, ax = plt.subplots()
        

        sns.boxplot(
            data=data,
            x="step",
            y="n",
            hue="step",
            showfliers=False,
            ax=ax
        )

        # sns.stripplot(
        #     data=data,
        #     x="step",
        #     y="n",
        #     marker="X",
        #     size=3,
        #     alpha=0.7,
        #     color="black"
        # )

        props = _outlier_proportion(data)
        y_max = data["n"].max()

        for i, step in enumerate(props.keys()):
            ax.text(
                i,
                0.92,
                props[step],
                ha="center",
                va="bottom",
                fontsize=9,
                transform=ax.get_xaxis_transform()
            )

        ax.set_title(f"Distribution of the number of coordinates of\n{folder}, thresh={likelihood_threshold:.2f}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Number of removed points")

        # plt.xticks(rotation=45)
        # plt.tight_layout()
        fig.savefig(make_output_path(cfg.paths.analysis,  f"{folder}_distri.png"))
        plt.close()


    all_data = pd.DataFrame(all_counts)

    fig, ax = plt.subplots()

    sns.boxplot(
        data=all_data,
        x="step",
        y="n",
        hue="step",
        showfliers=False,
        ax=ax
    )

    # sns.stripplot(
    #     data=all_data,
    #     x="step",
    #     y="n",
    #     marker="X",
    #     size=3,
    #     alpha=0.7,
    #     color="black"
    # )

    props = _outlier_proportion(all_data)
    y_max = all_data["n"].max()

    for i, step in enumerate(props.keys()):
        ax.text(
            i,
            0.92,
            props[step],
            ha="center",
            va="bottom",
            fontsize=9, 
            transform=ax.get_xaxis_transform()
        )

    ax.set_title(f"Distribution of the number of coordinates of {cfg.rat_name}\nlikelihood threshold={likelihood_threshold:.2f}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of removed points")

    # ax.set_ylim(-0.5, 20)

    # plt.xticks(rotation=45)
    # plt.tight_layout()
    fig.savefig(make_output_path(cfg.paths.analysis, f"{cfg.rat_name}_point_removal_distribution.png"))
    plt.close()



def _trial_report(cfg, trials: list[dict]) -> dict:

    def new_block(intensities):
        return {
            "Total": 0,
            "LaserOn": {"Total": 0, **{i: 0 for i in intensities}},
            "LaserOff": {"Total": 0, **{i: 0 for i in intensities}},
        }

    def init_group(intensities):
        return {
            "Total": 0,
            "Successful": new_block(intensities),
            "Unsuccessful": new_block(intensities),
            "Rejected": new_block(intensities),
            "No reward -": new_block(intensities),
            "No reward +": new_block(intensities),
            "No pad off": new_block(intensities),
            "No cue": new_block(intensities),
            "Unknown": new_block(intensities),
        }

    def update(group, outcome, laser_state, intensity):
        group[outcome]["Total"] += 1
        group[outcome][laser_state]["Total"] += 1
        group[outcome][laser_state][intensity] += 1


    report = {
        "Total number of trials": len(trials),
        "Beta": init_group(["1mW", "2,5mW"]),
        "Conti": init_group(["0,5mW", "0,75mW", "2,5mW"]),
        "NOstim": init_group(["NOstim"]),
    }

    #--------------------- loop ---------------------------

    for t in trials:
        condition = t["condition"]
        intensity = t["laser_intensity"]

        if "Beta" in condition:
            group = report["Beta"]
        elif "Conti" in condition:
            group = report["Conti"]
        else : 
            group = report["NOstim"]

        laser_state = t["laser_state"]


        # Successful
        if t[cfg.bodypart]["trial_success"]:
            group["Total"] += 1
            update(group, "Successful", laser_state, intensity)

            # No reward
            if not t["reward"]:
                update(group, "No reward +", laser_state, intensity)
        
        # Unsuccessful
        else : 
            group["Total"] += 1
            update(group, "Unsuccessful", laser_state, intensity)

            # No reward
            if not t["reward"]:
                update(group, "No reward -", laser_state, intensity)

            if not t["pad_off"] :
                update(group, "No pad off", laser_state, intensity)

            elif t["cue_type"] == "NoCue":
                update(group, "No cue", laser_state, intensity)

            elif t["pad_off"] and t["cue_type"] != "NoCue" :
                update(group, "Rejected", laser_state, intensity)

            else : 
                update(group, "Unknown", laser_state, intensity)
            

    return report



def _plot_trial_report(yaml_file: Path, output_path: Path):
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    trial_types = ["Beta", "Conti", "NOstim"]

    fig = make_subplots(
        rows=1,
        cols=len(trial_types),
        specs=[[{"type": "domain"}] * len(trial_types)],
        subplot_titles=[t for t in trial_types]
    )

    for i, trial_type in enumerate(trial_types, start=1):

        block = data[trial_type]

        labels = []
        parents = []
        values = []

        print()
        print(trial_type)
        print(f"Total: ", block["Total"])
        print(f"Successful total: ", block["Successful"]["Total"])
        print(f"Unsuccessful total: ", block["Unsuccessful"]["Total"])

        # ---- ROOT ----
        root_name = trial_type
        labels.append(root_name)
        parents.append("")
        values.append(block["Total"])

        # ---- SUCCESS ----
        labels.append("Successful")
        parents.append(root_name)
        values.append(block["Successful"]["Total"])

        # ---- UNSUCCESSFUL ----
        labels.append("Unsuccessful")
        parents.append(root_name)
        values.append(block["Unsuccessful"]["Total"])

        for reason in ["Rejected", "No pad off", "No cue", "Unknown"]:
            labels.append(reason)
            parents.append("Unsuccessful")
            values.append(block[reason]["Total"])

        fig.add_trace(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                textinfo="label+percent parent+value",
            ),
            row=1,
            col=i
        )

    fig.update_layout(
        title="Trial Outcomes with Failure Breakdown",
    )

    fig.write_html(str(output_path.with_suffix(".html")))
    fig.show()



def plot_trial_failure_reason(cfg, filenames) : 
    from rats_kinematics_utils.core.file_utils import load_trial_data

    output_dir = cfg.paths.analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "failure_reason_data.yaml"

    all_trials = []
    for i, metrics_path in enumerate(filenames) :
        metrics_path = Path(metrics_path) 
        print(metrics_path.stem)
        metrics = load_trial_data(metrics_path)
        for trial in metrics : 
            all_trials.append(trial)

    print("True number of trials :", len(all_trials))
    report = _trial_report(cfg, all_trials)

    # save report
    with open(output_dir / report_path, "w") as file :
        yaml.dump(report, file, default_flow_style=False, indent=4, sort_keys=False)


    # plot report
    print(f"Loading {report_path} and plotting")
    _plot_trial_report(output_dir / report_path,
                    output_dir / f"{cfg.rat_name}_failure_reason.png")