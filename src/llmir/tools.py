from pydantic import BaseModel
from typing import Any

class Tool(BaseModel):
    """
    A tool that can be used by the LLM.

    Attributes:
        name: The name of the tool.
        description: The description of the tool.
        input_schema: The schema of the input of the tool. A possible schema specification can be found at the Model Context Protocol Projekt: https://github.com/modelcontextprotocol
    """

    name: str
    description: str
    input_schema: dict[str, Any]

