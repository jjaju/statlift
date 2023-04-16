import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import json
from os.path import dirname, join
import re
from typing import Dict, Tuple


def infer_column_names(data: pd.DataFrame, column_definitions: Dict) -> Dict:
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
        _ = data["Date"]
        return column_definitions["ENG"]
    except Exception:
        pass
    try:
        _ = data["Datum"]
        return column_definitions["GER"]
    except Exception:
        pass

    raise Exception("Language of data not supported.")


def show_total_stats(data: pd.DataFrame) -> None:
    """Shows aggregated metrics accross all workouts and exercises.

    Args:
        data (pd.DataFrame): Pandas dataframe containing workout data.
    """
    total_workouts = len(pd.unique(data["workout_uid"]))
    total_sets = len(data[columns["REPS"]])
    total_reps = sum(data[columns["REPS"]])
    total_volume = sum(data["volume"])
    duration_data = data.groupby("workout_uid").agg(**{
        "duration": (columns["WORKOUT_DURATION"], "unique")
    })
    total_duration = sum(duration_data["duration"])
    cl1, cl2, cl3, cl4, cl5 = st.columns(5)
    cl1.metric(
        label="\# of Workouts",
        value="{:,}".format(int(total_workouts))
    )
    cl2.metric(
        label="Total Volume (kg)",
        value="{:,}".format(int(total_volume))
    )
    cl3.metric(
        label="\# of Sets",
        value="{:,}".format(int(total_sets))
    )
    cl4.metric(
        label="\# of Reps",
        value="{:,}".format(int(total_reps))
    )
    cl5.metric(
        label="\# of Minutes trained",
        value="{:,}".format(int(total_duration))
    )


if __name__ == "__main__":
    # setup page
    st.set_page_config(
        page_title="StatLift",
        page_icon=":mechanical_arm",
        layout="wide"
    )
    st.title("StatLift - Free Analytics for Strong Data :rocket:")
    st.write("[![Star](https://img.shields.io/github/stars/jjaju/statlift.svg?logo=github&style=social)](https://gitHub.com/jjaju/statlift)")
    
    # load csv file
    def on_upload() -> None:
        """deletes old data from session state upon csv upload."""
        st.session_state["data"] = None
        st.session_state["columns"] = None

    st.write("## :page_facing_up: Upload csv file (exported from Strong-App):")
    csv = st.file_uploader("_", label_visibility="hidden", on_change=on_upload)
    st.session_state["csv"] = csv
    if st.session_state["csv"] is None:
        exit()
        
    st.divider()
    
    # update data in session state if necessary
    if st.session_state["data"] is None:
        # load data
        data = pd.read_csv(st.session_state["csv"], sep=None, engine="python")
        json_path = join(dirname(__file__), "columns.json")
        with open(json_path) as f:
            column_definitions = json.load(f)
        columns = infer_column_names(data, column_definitions)

        # clean and prepare data
        data = data[[
            columns["DATE"],
            columns["WORKOUT_NAME"],
            columns["EXERCISE_NAME"],
            columns["WEIGHT"],
            columns["REPS"],
            columns["WORKOUT_DURATION"]
        ]]
        data.dropna(subset=[columns["WEIGHT"], columns["REPS"]], how='all', inplace=True)
        data[columns["WEIGHT"]] = data[columns["WEIGHT"]].fillna(0)
        data[columns["REPS"]] = data[columns["REPS"]].fillna(0)
        data[columns["DATE"]] = pd.to_datetime(data[columns["DATE"]]).dt.date
        data["workout_uid"] = data[columns["WORKOUT_NAME"]] + data[columns["DATE"]].copy().astype(str)
        data["volume"] = data[columns["WEIGHT"]] * data[columns["REPS"]]

        def convert_to_minutes(duration: str) -> int:
            """Converts workout duration from string representation to integers.

            Args:
                duration (str): Duration in the format "<hours>h <minutes>m" (e.g. 
                1h 20m)

            Returns:
                int: The corresponding number of minutes (e.g. "1h 20m" will return 
                80)
            """
            try:
                return 60 * int(re.findall(r"(\d+)h", duration)[0]) + int(re.findall(r"(\d+)m", duration)[0])
            except Exception:
                return 0

        data[columns["WORKOUT_DURATION"]] = data[columns["WORKOUT_DURATION"]].apply(convert_to_minutes)

        # save data and column names to session state
        st.session_state["data"] = data
        st.session_state["columns"] = columns
    
    # retrieve data and column names from session state
    data = st.session_state["data"]
    columns = st.session_state["columns"]
    
    # set date range
    st.write("## :date: Select date range:")
    fl1, fl2 = st.columns(2)
    start_date_filter = fl1.date_input("**Start date**", data[columns["DATE"]].min())
    end_date_filter = fl2.date_input("**End date**", data[columns["DATE"]].max())
    data = data[(data[columns["DATE"]] >= start_date_filter) & (data[columns["DATE"]] <= end_date_filter)]

    ###########################################################################
    # 1. overall metrics of workouts in date range
    ###########################################################################

    st.divider()
    st.write("## :bar_chart: Metrics across all workouts:")
    show_total_stats(data)

    ###########################################################################
    # 2. metrics and graphs for individual exercises
    ###########################################################################

    st.divider()
    st.write("## :mechanical_arm: Metrics for individual exercises:")

    exercise_filter = st.selectbox("**Select exercise**", pd.unique(data[columns["EXERCISE_NAME"]]))
    exercise_data = data[data[columns["EXERCISE_NAME"]] == exercise_filter].copy()
    exercise_data.loc[:, "workout_exercise_uid"] = (
        exercise_data[columns["WORKOUT_NAME"]] 
        + exercise_data[columns["EXERCISE_NAME"]] 
        + exercise_data[columns["DATE"]].copy().astype(str)
    )
    exercise_data = exercise_data.groupby("workout_exercise_uid").agg(**{
        "date": (columns["DATE"], "max"),
        "exercise": (columns["EXERCISE_NAME"], "unique"),
        "mean_reps": (columns["REPS"], "mean"),
        "max_weight": (columns["WEIGHT"], "max"),
        "max_reps": (columns["REPS"], "max"),
        "max_volume": ("volume", "max"),
        "total_volume": ("volume", "sum"),
        "total_reps": (columns["REPS"], "sum")
    })
    exercise_data["mean_weight"] = exercise_data["total_volume"] / exercise_data["total_reps"]
    
    prev_exercise_data = exercise_data.sort_values(by="date")
    prev_exercise_data = prev_exercise_data.iloc[:-1]

    # 2a. metrics
    st.write("##")
    st.write(f"##### :bar_chart: Metrics for *{exercise_filter}*:")
    ecl1, ecl2, ecl3, ecl4 = st.columns(4)
    
    def calculate_metric_and_delta(column: str, aggregation: str) -> Tuple[str, str]:
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
            metric = exercise_data[column].max()
            if prev_exercise_data[column].max() is not np.NaN:
                metric_prev = prev_exercise_data[column].max()
            else:
                metric_prev = metric
        elif aggregation == "sum":
            metric = exercise_data[column].sum()
            if prev_exercise_data[column].sum() is not np.NaN:
                metric_prev = prev_exercise_data[column].sum()
            else:
                metric_prev = metric
        elif aggregation == "len":
            metric = len(exercise_data[column])
            metric_prev = len(prev_exercise_data)
        else:
            raise Exception("Invalid aggregation method.")
        delta = metric - metric_prev
        metric = "{:,}".format(int(metric))
        delta = "{:,}".format(int(delta))
        return metric, delta

    total_sets, total_sets_delta = calculate_metric_and_delta("date", "len")
    total_reps, total_reps_delta = calculate_metric_and_delta("total_reps", "sum")
    total_volume, total_volume_delta = calculate_metric_and_delta("total_volume", "sum")
    max_weight, max_weight_delta = calculate_metric_and_delta("max_weight", "max")
    max_reps, max_reps_delta = calculate_metric_and_delta("max_reps", "max")
    max_volume, max_volume_delta = calculate_metric_and_delta("max_volume", "max")

    ecl1, ecl2, ecl3 = st.columns(3)
    ecl1.metric(
        label="Total Sets",
        value=total_sets,
        delta=total_sets_delta
    )
    ecl2.metric(
        label="Total Reps",
        value=total_reps,
        delta=total_reps_delta
    )
    ecl3.metric(
        label="Total Volume (kg)",
        value=total_volume,
        delta=total_volume_delta
    )
    ecl1.metric(
        label="Max Weight (kg)",
        value=max_weight,
        delta=max_weight_delta
    )
    ecl2.metric(
        label="Max Reps",
        value=max_reps,
        delta=max_reps_delta
    )
    ecl3.metric(
        label="Max Volume (kg)",
        value=max_volume,
        delta=max_volume_delta
    )

    # 2b. graphs
    st.write("##")
    st.write(f"##### :chart_with_upwards_trend: Graphs for *{exercise_filter}*:")
 
    metric_to_column = {
        "Total Volume (per workout)": "total_volume",
        "Mean Weight (across sets per workout)": "mean_weight",
        "Mean Reps (across sets per workout)": "mean_reps",
        "Max Weight (across sets per workout)": "max_weight",
        "Max Reps (across sets per workout)": "max_reps",
        "Max Volume (across sets per workout)": "max_volume",
    }
    selected_metrics = st.multiselect(
        "**Select metrics to plot**", 
        metric_to_column.keys(),
        default=metric_to_column.keys()
    )
    graph_columns = st.columns(2)

    for m, metric in enumerate(selected_metrics):
        col_index = m % 2
        col = graph_columns[col_index]
        chart = alt.Chart(
            exercise_data, title=f"{metric} for {exercise_filter}"
        ).mark_line(point=True).encode(
            x=alt.X("date", title="Date"),
            y=alt.Y(metric_to_column[metric], title=metric),
        )
        chart += chart.transform_regression('date', metric_to_column[metric]).mark_line(color="red")
        col.altair_chart(chart, use_container_width=True)

    ###########################################################################
    # 3. metrics and graphs for individual workout routines (e.g. pull day)
    ###########################################################################

    st.divider()
    st.write("## :repeat: Metrics for individual workout routines:")
    workout_filter = st.selectbox("**Select workout routine**", pd.unique(data[columns["WORKOUT_NAME"]]))
    data = data[data[columns["WORKOUT_NAME"]] == workout_filter]
    
    # 3a. metrics
    st.write("##")
    st.write(f"##### :bar_chart: Metrics for workout routine *{workout_filter}*:")
    show_total_stats(data)

    # 3b. graphs
    st.write("##")
    st.write(f"##### :chart_with_upwards_trend: Graphs for *{workout_filter}*:")
    workout_data = data.groupby("workout_uid").agg(**{
        "date": (columns["DATE"], "max"),
        "total_volume": ("volume", "sum"),
        "total_reps": (columns["REPS"], "sum")
    })
    metric_to_column_workout = {
        "Total Volume (per workout)": "total_volume",
        "Total Reps (per workout)": "total_reps"
    }
    graph_columns_workout = st.columns(2)
    for m, metric in enumerate(metric_to_column_workout.keys()):
        col_index = m % 2
        col = graph_columns_workout[col_index]
        chart = alt.Chart(
            workout_data, title=f"{metric} for {workout_filter}"
        ).mark_line(point=True).encode(
            x=alt.X("date", title="Date"),
            y=alt.Y(metric_to_column_workout[metric], title=metric),
        )
        chart += chart.transform_regression('date', metric_to_column_workout[metric]).mark_line(color="red")
        col.altair_chart(chart, use_container_width=True)
