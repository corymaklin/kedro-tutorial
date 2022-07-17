"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from adult.pipelines.adult import create_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    adult_pipeline = create_pipeline()

    return {"__default__": adult_pipeline}
