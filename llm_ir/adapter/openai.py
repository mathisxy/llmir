from typing import TypedDict, Literal, Union

from ..messages import AIMessage
from ..chunks import AIChunk, AIChunkText, AIChunkImageURL, AIChunkFile
import base64


class OpenAITextContent(TypedDict):
    type: Literal["text"]
    text: str


class OpenAIImageURLContent(TypedDict):
    type: Literal["image_url"]
    image_url: dict[str, str]

OpenAIContent = Union[OpenAITextContent, OpenAIImageURLContent]


class OpenAIMessage(TypedDict):
    role: str
    content: list[OpenAIContent]


def to_openai(messages: list[AIMessage]) -> list[OpenAIMessage]:
    

    result: list[OpenAIMessage] = []
    for message in messages:
        role = message.role.value
        result.append(OpenAIMessage(
            role= role,
            content= [
                chunk_to_openai(chunk) for chunk in message.chunks
            ]
        ))
    return result


def chunk_to_openai(chunk: AIChunk) -> OpenAIContent:

    match chunk:
        case AIChunkText():
            return OpenAITextContent(
                type="text",
                text=chunk.text,
            )
        case AIChunkImageURL():
            return OpenAIImageURLContent(
                type="image_url",
                image_url={
                    "url": chunk.url,
                }
            )
        case AIChunkFile():
            if chunk.mimetype.startswith("image/"):
                base64_data = base64.b64encode(chunk.bytes).decode('utf-8')
                return OpenAIImageURLContent(
                    type= "image_url",
                    image_url= {
                        "url": f"data:{chunk.mimetype};base64,{base64_data}",
                    }
                )
            else:
                raise ValueError(f"Unsupported file type for OpenAI: {chunk.mimetype}")
        case _:
            raise ValueError(f"Unsupported chunk type: {type(chunk)}")