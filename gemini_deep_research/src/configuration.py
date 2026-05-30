from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import Any, Optional

import os

class Configuration(BaseModel):
    """The Configuration for the agent."""

    query_generator_model: str = Field(
        default="qwen3",
        metadata={
            "description":"The name of the language model to use for the agent's query generation."
        }
    )
    query_generator_base_url: str = Field(
        default="http://.../v1",
        metadata={
            "description":"The base URL of the language model to use for the agent's query generation."
        }
    )

    reflection_model: str = Field(
        default="qwen3",
        metadata={
            "description":"The name of the language model to use for the agent's reflection."
        }
    )
    reflection_base_url: str = Field(
        default="http://.../v1",
        metadata={
            "description":"The base URL of the language model to use for the agent's reflection."
        }
    )

    answer_model: str = Field(
        default="qwen3",
        metadata={
            "description":"The name of the language model to use for the agent's answer."
        }
    )
    answer_base_url: str = Field(
        default="http://.../v1",
        metadata={
            "description":"The base URL of the language model to use for the agent's answer."
        }
    )

    number_of_initial_queries: int = Field(
        default=3,
        metadata={
            "description": "The number of initial search queries to generate."
        }
    )
    max_research_loops: int = Field(
        default=2,
        metadata={
            "description": "The maximum number of research loops to perform."
        }
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        Configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        raw_values: dict[str, any] = {
            name: os.environ.get(name.upper(), config.get(name))
            for name in cls.model_fields.keys()
        }

        values = {k:v for k, v in raw_values.items() if v is not None}

        return cls(**values)
    
