import pandas as pd
import numpy as np
import json
from typing import Dict, Tuple
import re
import streamlit as st
import datetime as dt


class SePump:
    """Class for wrangling workout data."""

    def __init__(self):
        """Initializes member dataframes"""
        self.data = None
        self.exercise_data = None
        self.prev_exercise_data = None
        self.workout_data = None
        self.workout_data_agg = None
        self.columns = None

    def load_data(self, csv: st.runtime.uploaded_file_manager.UploadedFile) -> None:
        """Loads data from csv into dataframe.

        Args:
            csv (st.runtime.uploaded_file_manager.UploadedFile): Uploaded csv 
            file.
        """
        self.data = pd.read_csv(csv, sep=None, engine="python")

    def load_column_names(self, column_definitions_path: str) -> None:
        """Retrieves applicable column names based on given dataframe.

        Args:
            column_definitions_path (str): Path to json file with column name 
            definitions.
        """
        with open(column_definitions_path, encoding='utf8') as f:
            column_definitions = json.load(f)
        self.columns = self.__infer_column_names(self.data, column_definitions)

    def __infer_column_names(self, data: pd.DataFrame, column_definitions: Dict) -> Dict:
        """Infers the dataframe's language and returns the corresponging column names.

        Args:
            data (pd.DataFrame): Pandas dataframe containing workout data.
            column_definitions (Dict): Dictionary of column names in different 
                languages. Currently supported: German & English

        Raises:
            Exception: Raised if an unsupported language is detected.

        Returns:
            Dict: Applicable mapping of column names.
        """
        try:
            _ = data["Duration"]
            return column_definitions["ENG_IOS"]
        except Exception:
            pass
        try:
            _ = data["Workout Duration"]
            return column_definitions["ENG_ANDROID"]
        except Exception:
            pass
        try:
            _ = data["Dauer"]
            return column_definitions["GER_IOS"]
        except Exception:
            pass
        try:
            _ = data["Workout-Dauer"]
            return column_definitions["GER_ANDROID"]
        except Exception:
            pass
        
        raise Exception("Language of data not supported.")

    def clean_data(self) -> None:
        """Performs initial data cleaning of given workout data."""
        self.data = self.data.drop_duplicates(keep="first")
        self.data = self.data[[
            self.columns["DATE"],
            self.columns["WORKOUT_NAME"],
            self.columns["EXERCISE_NAME"],
            self.columns["WEIGHT"],
            self.columns["REPS"],
            self.columns["WORKOUT_DURATION"],
            self.columns["NOTES"]
        ]]
        self.data.dropna(subset=[self.columns["WEIGHT"], self.columns["REPS"]], how='all', inplace=True)
        self.data[self.columns["WEIGHT"]] = self.data[self.columns["WEIGHT"]].fillna(0)
        self.data[self.columns["REPS"]] = self.data[self.columns["REPS"]].fillna(0)
        # hacky way of dealing with differently formatted decimal numbers, assuming nobody goes beyond 1000 kg
        self.data[self.columns["WEIGHT"]] = self.data[self.columns["WEIGHT"]].replace(",", ".", regex=True).astype(np.single)
        self.data[self.columns["REPS"]] = self.data[self.columns["REPS"]].replace(",", ".", regex=True).astype(np.single)
        self.data["workout_uid"] = (
            self.data[self.columns["WORKOUT_NAME"]]
            + self.data[self.columns["DATE"]].copy().astype(str)
            + self.data[self.columns["WORKOUT_DURATION"]]
        )
        self.data[self.columns["DATE"]] = pd.to_datetime(self.data[self.columns["DATE"]]).dt.date
        self.data["volume"] = self.data[self.columns["WEIGHT"]] * self.data[self.columns["REPS"]]
        self.data[self.columns["WORKOUT_DURATION"]] = self.data[self.columns["WORKOUT_DURATION"]].apply(
            self.__convert_duration_to_minutes
        )

    def update_date_range(self, start_date: dt.date, end_date: dt.date) -> None:
        """Updates workout data based on given start and end date.

        Args:
            start_date (dt.date): Date after which workouts are included.
            end_date (dt.date): Date before which workouts are included.
        """
        self.data = self.data[
            (self.data[self.columns["DATE"]] >= start_date)
            & (self.data[self.columns["DATE"]] <= end_date)
        ]

    def update_exercise_data(self, exercise: str) -> None:
        """Updates single exercise data based on given exercise name.

        Args:
            exercise (str): Name of the exercise.
        """
        exercise_data = self.data[self.data[self.columns["EXERCISE_NAME"]] == exercise].copy()
        exercise_data.loc[:, "workout_exercise_uid"] = (
            exercise_data[self.columns["WORKOUT_NAME"]]
            + exercise_data[self.columns["EXERCISE_NAME"]]
            + exercise_data[self.columns["DATE"]].copy().astype(str)
        )
        self.exercise_data = exercise_data.groupby("workout_exercise_uid").agg(**{
            "date": (self.columns["DATE"], "max"),
            "exercise": (self.columns["EXERCISE_NAME"], "first"),
            "mean_reps": (self.columns["REPS"], "mean"),
            "max_weight": (self.columns["WEIGHT"], "max"),
            "max_reps": (self.columns["REPS"], "max"),
            "max_volume": ("volume", "max"),
            "total_volume": ("volume", "sum"),
            "total_reps": (self.columns["REPS"], "sum"),
            "notes": (self.columns["NOTES"], "first")
        })
        self.exercise_data["mean_weight"] = self.exercise_data["total_volume"] / self.exercise_data["total_reps"]
        self.prev_exercise_data = self.exercise_data.sort_values(by="date")
        self.prev_exercise_data = self.prev_exercise_data.iloc[:-1]

    def calculate_exercise_metric_and_delta(self, column: str, aggregation: str) -> Tuple[str, str]:
        """Perfoms a certain aggregation of a given column of exercise data and 
            calculates the difference between the last two workouts.

        Args:
            column (str): Name of the column.
            aggregation (str): Aggregation method. Can be one of  [max, sum, 
                len]

        Raises:
            Exception: If not supported aggregation method is provided.

        Returns:
            Tuple[str, str]: (Result of aggregation, Delta)
        """
        if aggregation == "max":
            metric = self.exercise_data[column].max()
            if self.prev_exercise_data[column].max() is not np.NaN:
                metric_prev = self.prev_exercise_data[column].max()
            else:
                metric_prev = metric
        elif aggregation == "sum":
            metric = self.exercise_data[column].sum()
            if self.prev_exercise_data[column].sum() is not np.NaN:
                metric_prev = self.prev_exercise_data[column].sum()
            else:
                metric_prev = metric
        elif aggregation == "len":
            metric = len(self.exercise_data[column])
            metric_prev = len(self.prev_exercise_data)
        else:
            raise Exception("Invalid aggregation method.")
        delta = metric - metric_prev
        metric = "{:,}".format(int(metric))
        delta = "{:,}".format(int(delta))
        return metric, delta

    def update_workout_data(self, workout_name: str) -> None:
        """Updates single workout routine data based on given workout name.

        Args:
            workout_name (str): Name of the workout routine.
        """
        self.workout_data = self.data[self.data[self.columns["WORKOUT_NAME"]] == workout_name]

    def update_workout_data_agg(self) -> None:
        """Updates aggregated metrics for single workout routine."""
        self.workout_data_agg = self.workout_data.groupby("workout_uid").agg(**{
            "date": (self.columns["DATE"], "max"),
            "total_volume": ("volume", "sum"),
            "total_reps": (self.columns["REPS"], "sum")
        })

    def __convert_duration_to_minutes(self, duration: str) -> int:
        """Converts workout duration from string representation to integers.

        Args:
            duration (str): Duration in the format "<hours>h <minutes>m",
            "<hours>h", or "<minutes>m" (e.g. "1h 20m", "3h", or "30m").

        Returns:
            int: The corresponding number of minutes (e.g. "1h 20m" will return
            80).
        
        Raises:
            ValueError: If the duration format is invalid.
        """
        # Regex to match hours and minutes
        match = re.match(r'^(?:(\d+)h)?\s*(?:(\d+)m)?', duration)
        hours = match.group(1)
        minutes = match.group(2)

        if not match or not (hours or minutes):
            raise ValueError("Invalid duration format.")

        total_minutes = 0
        if hours:
            total_minutes += int(hours) * 60
        if minutes:
            total_minutes += int(minutes)
        
        return total_minutes
