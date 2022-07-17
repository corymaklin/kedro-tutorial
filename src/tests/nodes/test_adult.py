import json

import pytest
import os
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from src.adult.nodes.adult import clean

WORK_DIR = Path(os.path.abspath(__file__))


@pytest.fixture
def adult_records():
    with open(os.path.join(WORK_DIR.parent.parent, "resources/records/adult.json")) as json_file:
        json_records = json.load(json_file)
        return json_records


def test_clean(adult_records):
    input_df = pd.DataFrame([
        {
            **adult_records[0],
            "workclass": "?"
        },
        {
            **adult_records[1],
            "workclass": "Self-emp-not-inc"
        }
    ])

    actual_df = clean(input_df)

    expected_df = pd.DataFrame([
        {
            **adult_records[1],
            "workclass": "Self-emp-not-inc"
        }
    ])

    assert_frame_equal(actual_df.reset_index(drop=True), expected_df.reset_index(drop=True))
