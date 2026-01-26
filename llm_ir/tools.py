from pydantic import BaseModel, Field
from typing import Any, Callable, Literal, Union

class ToolBase(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]

class MCPTool(ToolBase):
    type: Literal['mcp'] = "mcp"
    server_id: str

class FunctionTool(ToolBase):
    type: Literal['function'] = "function"
    function: Callable[..., Any] = Field(exclude=True)


Tool = Union[MCPTool, FunctionTool]

