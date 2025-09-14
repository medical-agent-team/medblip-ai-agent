#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MedBLIP model utilities shared across app and tests.

Pure helpers only; no Streamlit dependencies or side effects.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from transformers import BlipForConditionalGeneration, BlipProcessor


DEFAULT_MODEL_PATHS = (
    "./model",     # Local development
    "/app/model",  # Docker container path
)


def find_model_path(paths: tuple[str, ...] = DEFAULT_MODEL_PATHS) -> Optional[str]:
    """Return the first existing model path or None."""
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_medblip_model(
    *,
    model_path: Optional[str] = None,
    local_files_only: bool = True,
) -> Tuple[Optional[BlipForConditionalGeneration], Optional[BlipProcessor], Optional[str]]:
    """Load finetuned MedBLIP model and processor if available locally.

    Returns (model, processor, resolved_path); any may be None when not found.
    """
    resolved = model_path or find_model_path()
    if not resolved:
        return None, None, None

    try:
        model = BlipForConditionalGeneration.from_pretrained(
            resolved, local_files_only=local_files_only
        )
        processor = BlipProcessor.from_pretrained(
            resolved, local_files_only=local_files_only
        )
        return model, processor, resolved
    except Exception:
        # Keep callers robust; they can handle None to show offline/demo paths
        return None, None, resolved

