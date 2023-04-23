import streamlit as st


def v_space(lines: int) -> None:
    """Super hacky solution to add vertical white space.

    Args:
        lines (int): Number of lines of white space.
    """
    for _ in range(lines):
        st.write('&nbsp;')
