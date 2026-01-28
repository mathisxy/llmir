from typing import TypedDict, Literal

from ..messages import AIMessages, AIMessageToolResponse
from ..chunks import AIChunkText, AIChunkImageURL, AIChunkFile, AIChunkToolCall
import base64
import json


class OpenAIText(TypedDict):
    type: Literal["text"]
    text: str

class OpenAIImageURLURL(TypedDict):
    url: str

class OpenAIImageURL(TypedDict):
    type: Literal["image_url"]
    image_url: OpenAIImageURLURL


class OpenAIToolCallFunction(TypedDict):
    name: str
    arguments: str

class OpenAIToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: OpenAIToolCallFunction

    

OpenAIContents = OpenAIText | OpenAIImageURL


class OpenAIMessage(TypedDict):
    role: str
    content: list[OpenAIContents]
    tool_calls: list[OpenAIToolCall]

class OpenAIMessageToolResponse(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    content: str


OpenAIMessages = OpenAIMessage | OpenAIMessageToolResponse



def to_openai(messages: list[AIMessages]) -> list[OpenAIMessages]:
    

    result: list[OpenAIMessages] = []
    for message in messages:
        role = message.role.value
        if isinstance(message, AIMessageToolResponse):
            assert(role == "tool")
            text: str = ""
            media_chunks: list[AIChunkFile | AIChunkImageURL] = []
            for chunk in message.chunks:
                if isinstance(chunk, AIChunkText):
                    text += chunk.text
                elif isinstance(chunk, AIChunkImageURL) or isinstance(chunk, AIChunkFile):
                    media_chunks.append(chunk)
                else:
                    raise Exception(f"Invalid chunk type for Tool Response: {chunk.type}")
            result.append(
                OpenAIMessageToolResponse(
                    role=role,
                    tool_call_id=message.id,
                    content=text,
                )
            )
            if media_chunks:
                result.append(
                    OpenAIMessage(
                        role="user", # Hacky, but what else to circumvent API limitations in a broadly compatible way?
                        content=[
                            content_chunk_to_openai(chunk) for chunk in media_chunks
                        ],
                        tool_calls=[]
                    )
                )
        else:
            result.append(OpenAIMessage(
                role= role,
                content=[
                    content_chunk_to_openai(chunk) for chunk in message.chunks if not isinstance(chunk, AIChunkToolCall)
                ],
                tool_calls=[
                    tool_call_chunk_to_openai(chunk) for chunk in message.chunks if isinstance(chunk, AIChunkToolCall)
                ]
            ))
    return result


def content_chunk_to_openai(chunk: AIChunkText | AIChunkFile | AIChunkImageURL) -> OpenAIContents:

    match chunk:
        case AIChunkText():
            return OpenAIText(
                type="text",
                text=chunk.text,
            )
        case AIChunkImageURL():
            return OpenAIImageURL(
                type="image_url",
                image_url={
                    "url": chunk.url,
                }
            )
        case AIChunkFile():
            if chunk.mimetype.startswith("image/"):
                base64_data = base64.b64encode(chunk.bytes).decode('utf-8')
                return OpenAIImageURL(
                    type= "image_url",
                    image_url= {
                        "url": f"data:{chunk.mimetype};base64,{base64_data}",
                    }
                )
            elif chunk.mimetype == "text/plain":
                text = chunk.bytes.decode(encoding="utf-8")
                return OpenAIText(
                    type="text",
                    text=text
                )
            else:
                raise ValueError(f"Unsupported file type for OpenAI: {chunk.mimetype}")
        case _:
            raise ValueError(f"Unsupported chunk type: {type(chunk)}")
        
def tool_call_chunk_to_openai(chunk: AIChunkToolCall) -> OpenAIToolCall:

    return OpenAIToolCall(
                id=chunk.id,
                type="function",
                function=OpenAIToolCallFunction(
                    name=chunk.name,
                    arguments=json.dumps(chunk.arguments)
                )
            )