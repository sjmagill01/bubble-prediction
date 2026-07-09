"""
wrds_utils.py — WRDS connection helper.

Requires a WRDS account (https://wrds-www.wharton.upenn.edu/).
Set the WRDS_USERNAME environment variable, or let the wrds package
prompt for credentials on first use (it can cache them in ~/.pgpass).
"""
import os

import wrds


def _connect() -> "wrds.Connection":
    """Connect to WRDS using WRDS_USERNAME env var if set."""
    username = os.environ.get("WRDS_USERNAME")
    if username:
        return wrds.Connection(wrds_username=username)
    return wrds.Connection()
