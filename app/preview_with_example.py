"""Open the application with the bundled example routes already loaded.

Route search does not run on this machine, and without routes the review tabs
stay hidden. This seeds the session with the example payload that ships in the
repository, then runs app.py unchanged.

    streamlit run preview_with_example.py --server.port 8530

The routes are software-test material, not a synthesis proposal. This file is
a viewer; app.py is still the entry point for real use.
"""

from pathlib import Path
import runpy

import streamlit as st

import engine
import workspace as context

if "review_workspace" not in st.session_state:
    payload = engine.load_example()
    target = payload["routes"][0]["target"]
    state = context.load_routes(context.new_workspace(target), payload)
    st.session_state["review_workspace"] = state
    st.session_state["target_smiles"] = target
    st.session_state["target_catalog"] = "직접 입력"
    st.session_state["inspection_mode"] = "route"

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
