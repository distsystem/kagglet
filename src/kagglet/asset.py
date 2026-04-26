"""Shared Kaggle asset identity types."""

import pydantic


def split_slug(slug: str) -> tuple[str, str]:
    parts = slug.split("/", 1)
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[1]


class KaggleAsset(pydantic.BaseModel):
    """Base identity for Kaggle assets addressed as `{owner}/{name}`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    owner: str = ""
    name: str
    version: int = pydantic.Field(default=0, repr=False)

    @pydantic.model_validator(mode="before")
    @classmethod
    def parse_slug(cls, data):
        if not isinstance(data, dict) or "slug" not in data:
            return data
        data = dict(data)
        slug = data.pop("slug")
        if "name" not in data:
            data["owner"], data["name"] = split_slug(str(slug))
        return data

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other

    @property
    def ref(self) -> str:
        return f"{self.owner}/{self.name}" if self.owner else self.name

    @property
    def slug(self) -> str:
        return self.ref

    @property
    def versioned_slug(self) -> str:
        return f"{self.slug}/{self.version}"
