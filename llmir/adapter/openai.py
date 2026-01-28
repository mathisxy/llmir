from typing import TypedDict, Literal, Union

from ..messages import AIMessages, AIMessageToolResponse
from ..chunks import AIChunk, AIChunkText, AIChunkImageURL, AIChunkFile, AIChunkToolCall
import base64
import json


class OpenAITextContent(TypedDict):
    type: Literal["text"]
    text: str

class OpenAIImageURLURL(TypedDict):
    url: str

class OpenAIImageURLContent(TypedDict):
    type: Literal["image_url"]
    image_url: OpenAIImageURLURL


class OpenAIToolCallFunction(TypedDict):
    name: str
    arguments: str

class OpenAIToolCallContent(TypedDict):
    id: str
    type: Literal["function"]
    function: OpenAIToolCallFunction

    

OpenAIContent = Union[OpenAITextContent, OpenAIImageURLContent, OpenAIToolCallContent]


class OpenAIMessage(TypedDict):
    role: str
    content: list[OpenAIContent]

class OpenAIMessageToolResponse(OpenAIMessage):
    tool_call_id: str
    name: str


def to_openai(messages: list[AIMessages]) -> list[OpenAIMessage]:
    

    result: list[OpenAIMessage] = []
    for message in messages:
        role = message.role.value
        if isinstance(message, AIMessageToolResponse):
            result.append(
                OpenAIMessageToolResponse(
                    role=role,
                    tool_call_id=message.id,
                    name=message.name,
                    content=[
                        chunk_to_openai(chunk) for chunk in message.chunks
                    ]
                )
            )
        else:
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
            elif chunk.mimetype == "text/plain":
                text = chunk.bytes.decode(encoding="utf-8")
                return OpenAITextContent(
                    type="text",
                    text=text
                )
            else:
                raise ValueError(f"Unsupported file type for OpenAI: {chunk.mimetype}")
        case AIChunkToolCall():
            return OpenAIToolCallContent(
                id=chunk.id,
                type="function",
                function=OpenAIToolCallFunction(
                    name=chunk.name,
                    arguments=json.dumps(chunk.arguments)
                )
            )
        case _:
            raise ValueError(f"Unsupported chunk type: {type(chunk)}")