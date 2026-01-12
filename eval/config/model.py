from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import Field

from eval.types import ComponentImpl


class CommercialModelConfig(ComponentImpl):
    kind: Literal["commercial"] = "commercial"
    provider: str = '' 
    name: str = ''
    comment: str = '' 

    api_key: str = ''
    base_url: str = ''

class PublicModelConfig(ComponentImpl):
    kind: Literal["public"] = "public"

    name: str = ''
    checkpoint_path: Path = Path('')

ModelUnion = Annotated[
    Union[CommercialModelConfig, PublicModelConfig],
    Field(discriminator="kind")
]