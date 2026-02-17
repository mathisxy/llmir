from typing import Union, Literal
from pydantic import BaseModel
from .rich import RichReprMixin

class AIChunkText(RichReprMixin, BaseModel):
    """
    A text chunk in the LLM conversation.

    Attributes:
        type: The type of the chunk, only for discriminated unions.
        text: The text content.
    """

    type: Literal["text"] = "text"
    text: str

class AIChunkFile(RichReprMixin, BaseModel):
    """
    A file chunk in the LLM conversation.
    
    The file is represented as bytes.
    It does not contain any pointer to the file system.
    
    Attributes:
        type: The type of the chunk, only for discriminated unions.
        name: The name of the file.
        mimetype: The mimetype of the file.
        bytes: The bytes of the file.
    """

    type: Literal["file"] = "file"
    name: str
    mimetype: str
    bytes: bytes

class AIChunkImageURL(RichReprMixin, BaseModel):
    """
    An image chunk in the LLM conversation.

    The image is represented as an URL that points to the image data.

    Attributes:
        type: The type of the chunk, only for discriminated unions.
        url: The URL of the image.
    """

    type: Literal["image"] = "image"
    url: str

class AIChunkToolCall(RichReprMixin, BaseModel):
    """
    A tool call chunk in the LLM conversation.

    Attributes:
        type: The type of the chunk, only for discriminated unions.
        id: The id of the tool.
        name: The name of the tool.
        arguments: The arguments of the tool call. The arguments are a dictionary of key-value pairs. The values can be of any type, but should be JSON serializable. The keys are the names of the arguments.
    """


    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, object]


AIChunks = Union[AIChunkText, AIChunkFile, AIChunkImageURL, AIChunkToolCall]