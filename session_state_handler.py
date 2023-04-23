import streamlit as st
from typing import List


def on_csv_upload():
    set_true(["updated_csv", "updated_date", "updated_exercise", "updated_workout"])


def on_date_change():
    set_true(["updated_date", "updated_exercise", "updated_workout"])


def on_exercise_change():
    set_true(["updated_exercise"])


def on_workout_change():
    set_true(["updated_workout"])


def init_session_state_updates():
    set_false(["updated_csv", "updated_date", "updated_exercise", "updated_workout"])


def set_true(entries: List) -> None:
    for entry in entries:
        st.session_state[entry] = True


def set_false(entries: List) -> None:
    for entry in entries:
        st.session_state[entry] = False
