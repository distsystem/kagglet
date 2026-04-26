"""Kaggle kernel asset metadata."""

import enum

import pydantic

from kagglet.asset import KaggleAsset
from kagglet.api.meta import KernelMeta


class Accelerator(enum.StrEnum):
    NONE = ""
    GPU_H100 = "H100"
    GPU_P100 = "P100"
    GPU_T4_X2 = "T4x2"
    TPU_V5E8 = "TpuV5E8"

    @property
    def uses_gpu(self) -> bool:
        return bool(self.value) and not self.uses_tpu

    @property
    def uses_tpu(self) -> bool:
        return self.value.lower().startswith("tpu")

    @property
    def machine_shape(self) -> str | None:
        return self.value or None


class KaggleKernel(KaggleAsset):
    """Kaggle kernel resource metadata."""

    title: str = ""
    dataset_sources: list[str] = pydantic.Field(default_factory=list)
    model_sources: list[str] = pydantic.Field(default_factory=list)
    internet: bool = True
    competition: str = ""
    accelerator: Accelerator = Accelerator.NONE

    @property
    def display_title(self) -> str:
        return self.title or self.name.replace("-", " ").replace("_", " ")

    @property
    def metadata(self) -> KernelMeta:
        return KernelMeta(
            id=self.slug,
            title=self.display_title,
            enable_gpu=str(self.accelerator.uses_gpu).lower(),
            enable_tpu=str(self.accelerator.uses_tpu).lower(),
            machine_shape=self.accelerator.machine_shape,
            enable_internet=str(self.internet).lower(),
            dataset_sources=list(self.dataset_sources),
            competition_sources=[self.competition] if self.competition else [],
            model_sources=self.model_sources or None,
        )
