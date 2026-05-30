from typing import Optional, List, Dict, Literal
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from enum import Enum

class step(BaseModel):
    title: str = ""
    description: str = ""
    status: Literal["pending", "completed"] = "pending"

class Plan(BaseModel):
    goal: str = ""
    thought: str = ""
    steps: List[step] = []

class State(MessagesState):
    user_message: str = ""
    plan: Plan
    observations: List = []
    final_report: str = ""
