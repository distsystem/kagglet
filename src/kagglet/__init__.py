"""Kagglet — automation toolkit for Kaggle."""

from kagglet.api import Kaggle as Kaggle, kaggle as kaggle
from kagglet.assets import (
    KAGGLE_INPUT as KAGGLE_INPUT,
    Accelerator as Accelerator,
    KaggleAsset as KaggleAsset,
    KaggleModel as KaggleModel,
    KaggleKernel as KaggleKernel,
    KaggleDataset as KaggleDataset,
)
from kagglet.notebook import NotebookProject as NotebookProject, percent_to_notebook as percent_to_notebook
