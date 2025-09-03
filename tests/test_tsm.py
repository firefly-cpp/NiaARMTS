import pytest
import os
import pandas as pd
import numpy as np
from niaarmts import Dataset
from niaarmts.NiaARMTS import NiaARMTS
from niaarmts.metrics import calculate_timestamp_metric


def test_tsm_timestamp_10pct_window_september24():
    """TSM should be 0.9 when the segment is exactly 10% of total span (20% - 30%)."""
    df = pd.read_csv("datasets/september24.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    t0, tT = df["timestamp"].min(), df["timestamp"].max()
    start = t0 + (tT - t0) * 0.20
    end   = start + (tT - t0) * 0.10  # 10% wide segment

    tsm = calculate_timestamp_metric(df, start, end, use_interval=False)

    assert 0.0 <= tsm <= 1.0
    assert tsm == pytest.approx(0.9, abs=1e-12)


def test_tsm_interval_proportional_window_intervals_csv():
    """
    Interval-domain TSM: pick a 25% wide window defined relative to [min, max].
    Expected TSM = 1 - 0.25 = 0.75 regardless of absolute values.
    """
    df = pd.read_csv("datasets/intervals.csv")
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce")

    t0, tT = df["interval"].min(), df["interval"].max()
    start = t0 + (tT - t0) * 0.25     # begin at 25% of the span
    end   = t0 + (tT - t0) * 0.50     # end at 50% of the span (width = 25%)

    tsm = calculate_timestamp_metric(df, start, end, use_interval=True)
    assert 0.0 <= tsm <= 1.0
    assert tsm == pytest.approx(0.75, abs=1e-12)

def test_custom_start_end_tsm():
    df = pd.read_csv("datasets/september24.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    t0, tT = df["timestamp"].min(), df["timestamp"].max()

    start = pd.Timestamp("2024-09-17 00:59:54")
    end = pd.Timestamp("2024-09-17 01:33:14")

    tsm = calculate_timestamp_metric(df, start, end, use_interval=False)
    assert tsm == pytest.approx(0.998638, abs=1e-4)
