"""Pydantic schemas for Kaggle API JSON bodies."""

from pydantic import Field, BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelMeta(BaseModel):
    """Base for Kaggle metadata schemas that serialize as camelCase JSON."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")

    def to_json(self) -> str:
        return self.model_dump_json(by_alias=True)


class ModelMeta(CamelMeta):
    """`model-metadata.json` body for `kaggle models create new`."""

    owner_slug: str
    slug: str
    title: str
    is_private: bool = True
    description: str = ""


class InstanceMeta(CamelMeta):
    """`model-instance-metadata.json` body for `kaggle models instances create`."""

    owner_slug: str
    model_slug: str
    instance_slug: str
    framework: str
    license_name: str = "Apache 2.0"
    overview: str = ""
    usage: str = ""


class KernelMeta(BaseModel):
    """`kernel-metadata.json` body for `kaggle kernels push` (snake_case schema)."""

    model_config = ConfigDict(extra="forbid")

    code_file: str = "notebook.ipynb"
    language: str = "python"
    kernel_type: str = "notebook"
    is_private: str = "true"
    enable_gpu: str
    machine_shape: str | None
    id: str
    title: str
    enable_internet: str
    dataset_sources: list[str] = Field(default_factory=list)
    competition_sources: list[str] = Field(default_factory=list)
    kernel_sources: list[str] = Field(default_factory=list)
    model_sources: list[str] | None = None

    def to_json(self) -> str:
        # Drop `model_sources` when unset so we match the historical layout
        # (Kaggle accepts either, but the diff stays smaller).
        exclude = {"model_sources"} if self.model_sources is None else set()
        return self.model_dump_json(indent=2, exclude=exclude)
