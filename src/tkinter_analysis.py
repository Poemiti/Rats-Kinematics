from pathlib import Path
import os
import pandas as pd
import tkinter as tk
from tkinter import ttk

from video_database import classify_video

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view

        for column in model.filters.keys():
            values = model.get_unique_values(column)
            view.create_filter_group(
                column,
                values,
                self.on_filter_selected
            )

        view.show_results(model.get_filtered())

    def on_filter_selected(self, column, value):
        self.model.set_filter(column, value)
        self.view.highlight_selection(column, value)
        self.view.show_results(self.model.get_filtered())



class View(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video File Filter")
        self.geometry("1900x1000")
        self.frames = {}
        self.buttons = {}
        self.result_box = None

    def create_filter_group(self, name, values, callback):
        frame = ttk.LabelFrame(self, text=name)
        frame.pack(fill="x", padx=5, pady=5)

        self.buttons[name] = {}

        for val in values:
            btn = ttk.Button(
                frame,
                text=str(val),
                command=lambda v=val: callback(name, v)
            )
            btn.pack(side="left", padx=2)
            self.buttons[name][val] = btn

        self.frames[name] = frame

    def highlight_selection(self, group, selected_value):
        for val, btn in self.buttons[group].items():
            btn.state(["!pressed"])
            if val == selected_value:
                btn.state(["pressed"])

    def show_results(self, dataframe):
        if self.result_box:
            self.result_box.destroy()

        self.result_box = tk.Text(self, height=10)
        self.result_box.pack(fill="both", expand=True)

        self.result_box.insert("end", dataframe.to_string())



class Model:
    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe
        self.filters = {
            "rat_name": None,
            "condition": None,
            "stim_location": None,
            "handedness": None,
            "view": None,
        }

    def set_filter(self, key, value):
        self.filters[key] = value

    def get_unique_values(self, column):
        return sorted(self._df[column].unique())

    def get_filtered(self):
        df = self._df
        for key, value in self.filters.items():
            if value is not None:
                df = df[df[key] == value]
        return df



def make_database(root_dir):
    sorted_videos = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            classify_video(os.path.join(root, name), sorted_videos)
    return pd.DataFrame(sorted_videos)



if __name__ == "__main__" : 

    RAW_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")
    PREDICTION_DIR =  Path("../data/csv_results")

    DATABASE: pd.DataFrame = make_database(RAW_VIDEO_DIR)

    model = Model(DATABASE)
    view = View()
    controller = Controller(model, view)

    view.mainloop()
