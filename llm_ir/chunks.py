from typing import Union, Literal
from pydantic import BaseModel

class AIChunkText(BaseModel):
    type: Literal["text"] = "text"
    content: str

class AIChunkFile(BaseModel):
    type: Literal["file"] = "file"
    name: str
    mimetype: str
    content: bytes

class AIChunkImageURL(BaseModel):
    type: Literal["image"] = "image"
    url: str

AIChunk = Union[AIChunkText, AIChunkFile, AIChunkImageURL]