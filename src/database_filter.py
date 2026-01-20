from pathlib import Path
import os
import pandas as pd
import tkinter as tk
from tkinter import ttk

from sort_files import make_database, is_csv, is_video

class Controller:           # where we controle what does each button do 
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.filtered_dataset = None
        self.dataset_name = view.dataset_name_input()

        for column in model.filters.keys():
            values = model.get_unique_values(column)
            view.filter_buttons(
                column,
                values,
                self.on_filter_selected
            )

        view.save_button(self.on_save)
        view.reset_button(self.on_reset)
        view.show_results(model.get_filtered())

    def on_filter_selected(self, column, value):
        self.model.set_filter(column, value)
        self.view.highlight_selection(column, value)
        self.view.show_results(self.model.get_filtered())

    def on_save(self) : 
        self.filtered_dataset = self.model.get_filtered()
        self.view.destroy()

    def on_reset(self) : 
        self.model.reset_filters()
        self.view.clear_all_selections()
        self.view.show_results(self.model.get_filtered())



class View(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Video File Filter")
        self.geometry("1900x1000")
        self.frames = {}
        self.buttons = {}
        self.result_box = None
        
        self.info_frame = ttk.Frame(self)
        self.info_frame.pack(fill="x", padx=5, pady=5)

        self.result_frame = ttk.Frame(self)
        self.result_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.n_file_label = ttk.Label(self.info_frame, text="Files number = 0")
        self.n_file_label.pack(anchor="w")

    def save_button(self, callback) :
        btn = ttk.Button(
            self,
            text="save",
            command=callback
        )
        btn.pack(side="bottom", padx=5, pady=5)
        self.buttons["save"] = btn

    # add a reset button
    def reset_button(self, callback) :
        btn = ttk.Button(
            self,
            text="reset",
            command=callback
        )
        btn.pack(side="bottom", padx=5, pady=5)
        self.buttons["reset"] = btn

    # clear al the selection of every filter
    def clear_all_selections(self):
        for group in self.buttons:
            if group in ("save", "reset"):
                continue
            for btn in self.buttons[group].values():
                btn.state(["!pressed"])

    # input string var to rename the filtered dataset
    def dataset_name_input(self):
        frame = ttk.LabelFrame(self, text="Enter dataset name (for saving) :")
        frame.pack(fill="x", padx=5, pady=5)

        self.dataset_name = tk.StringVar()

        entry = ttk.Entry(frame, textvariable=self.dataset_name, width=40)
        entry.pack(side="left", padx=5, pady=5)

        return self.dataset_name


    def filter_buttons(self, name, values, callback):
        frame = ttk.LabelFrame(self, text=name)
        frame.pack(fill="both", padx=5, pady=5)

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

    def show_results(self, dataframe) :
        # Update file count
        self.n_file_label.config(text=f"Files number = {len(dataframe)}")

        # Replace result box only
        if self.result_box:
            self.result_box.destroy()

        self.result_box = tk.Text(self.result_frame, height=10)
        self.result_box.pack(fill="both", expand=True)

        self.result_box.insert("end", dataframe.to_string())
    



class Model:   # focus on filtration manipulation
    def __init__(self, dataframe: pd.DataFrame):
        self._df = dataframe
        self.filters = {column: None for column in dataframe.columns[1:]} 
        # add filter for every columns of the dataframe

    def set_filter(self, key, value):
        self.filters[key] = value

    def reset_filters(self) : 
        for key in self.filters : 
            self.filters[key] = None

    def get_unique_values(self, column):
        return sorted(self._df[column].unique())

    def get_filtered(self):
        df = self._df
        for key, value in self.filters.items():
            if value is not None:
                df = df[df[key] == value]
        return df



if __name__ == "__main__" : 

    DATABASE_DIR = Path("../data/database")
    RAW_VIDEO_DIR = Path("/media/filer2/T4b/Datasets/Rats/Photron_Video/Raphael2024")
    PREDICTION_DIR =  Path("../data/csv_results")
    CLIP_DIR = Path("../data/clips")

    DATABASE: pd.DataFrame = make_database(RAW_VIDEO_DIR, is_video)
    # DATABASE_PRED: pd.DataFrame = make_database(PREDICTION_DIR, is_csv)
    # DATABASE_CLIP : pd.DataFrame = make_database(CLIP_DIR, is_video)

    print(DATABASE)

    model = Model(DATABASE) # or DATABASE_PRED
    view = View()
    controller = Controller(model, view)

    view.mainloop()

    FILTERED_DATABASE = controller.filtered_dataset

    dataset_name = f"{controller.dataset_name.get().strip()}.csv"
    FILTERED_DATABASE.to_csv(DATABASE_DIR / dataset_name)
    print(f"Filtered dataset saved as : {DATABASE_DIR / dataset_name}")
    print(f"Number of files in {dataset_name} : {len(FILTERED_DATABASE)}")
