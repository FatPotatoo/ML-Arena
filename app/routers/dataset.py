"""
Dataset endpoints.

A "router" is just a group of related URLs. This one owns everything under
/api/dataset. Keeping each topic in its own router file stops main.py from
turning into one giant file.
"""
from fastapi import APIRouter

from ..data import dataset_info

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


@router.get("/info")
def get_info():
    """GET /api/dataset/info -> summary of the dataset, split, and features.

    The frontend calls this once on load to know which feature checkboxes to
    show and how big each split is.
    """
    return dataset_info()
