import json
from pydantic import BaseModel
from typing import List, Optional
import datetime


class ToolFnSchema(BaseModel):
    input: Optional[str]
    # document_type: Optional[str]
    # title: Optional[str]
    # authors: Optional[List[str]]
    # channel_name: Optional[str]
    # video_link: Optional[str]
    # pdf_link: Optional[str]
    # release_date: Optional[datetime.date]


def schema(model: BaseModel, title: str = "DefaultToolFnSchema", description: str = "Default tool function Schema.") -> str:
    # 1. Generate the schema
    schema_dict = model.model_json_schema()

    # 2. Format the schema to match the desired format
    formatted_schema = {
        "title": title,
        "description": description,
        "type": "object",
        "properties": schema_dict["properties"],
        "required": schema_dict.get("required", [])
    }

    # 3. Convert the schema dictionary to a string representation
    schema_str = json.dumps(formatted_schema, indent=4)  # Pretty-printed for clarity, you can remove indent if you want a compact string.
    return schema_str
