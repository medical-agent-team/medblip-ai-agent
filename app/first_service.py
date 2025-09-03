#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Thin wrapper to run the unified Streamlit app in app/main.py.
This preserves the existing `streamlit run app/first_service.py` workflow.
"""

from app.main import main as run_app

if __name__ == "__main__":
    run_app()

