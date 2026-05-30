from tools_schemas import SearchQueryList, Reflection, knowledge_search, web_search
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from configuration import Configuration
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph import END, START
from langgraph.types import Send
from google.genai import Client
from dotenv import load_dotenv

from state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)

from utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls
)

import json
import os

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

def extract_json(text):
    if "```json" not in text:
        return text
    text = text.split("```json")[1].split("```")[0].strip()
    return text

def extract_answer(text):
    if "<think>" in text:
        answer = text.split("<think>")[-1]
        return answer.strip()
    
    return text

def parse_tools(text, start_flag, end_flag):
    tools = text.split(start_flag)
    tools = [tool for tool in tools if end_flag in tool]
    if tools:
        tools = [tool.split(end_flag)[0].strip() for tool in tools]
    return tools

def get_tools(response):
    print(extract_answer(response["content"]))

    if response["tool_calls"]:
        print("----------------------------------")
        tools = response["tool_calls"]
    
    else:
        content = extract_answer(response["content"])

        if "<tool_call>" in content:
            print("-----------------<tool_call>-----------------")
            tools = parse_tools(content, "<tool_call>", "<tool_call>")
        elif "<function_call>" in content:
            print("-----------------<function_call>-----------------")
            tools = parse_tools(content, "<function_call>", "<function_call>")
        elif "```json\n[" in content:
            print("-----------------<```json>-----------------")
            tools = parse_tools(content, "```json\n[", "]```")
        elif "```json" in content and ("name" in content and ("args" in content or "arguments" in content)):
            print("-----------------<```json>-----------------")
            tools = parse_tools(content, "```json", "```")
        else:
            tools = []
    
    return tools

def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """
    LangGraph node that generates search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config=config)
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    
    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        base_url=configurable.query_generator_base_url,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"]
    )

    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}

def continue_to_web_research(state: QueryGenerationState):
    """
    LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]

def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """
    LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    configurable = Configuration.from_runnable_config(config=config)
    formatted_prompt = web_searcher_instructions.format(
        research_topic=state["search_query"]
    )

    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=configurable.query_generator_base_url
    )

    messages = [HumanMessage(content=formatted_prompt)]
    web_research_result = []

    tools = {"web_search": web_search, "knowledge_search": knowledge_search}
    while True:
        response = llm.bind_tools([web_search, knowledge_search]).invoke(messages)
        response = response.model_dump_json(indent=4, exclude_none=True)
        response = json.laods(response)
        extract_tools = get_tools(response=response)
        if extract_tools:
            for tool in extract_tools:
                if isinstance(tool, str):
                    try:
                        tool = json.loads(tool)
                    except Exception as e:
                        messages += [HumanMessage(content=f"{tool}json格式错误:{e}")]
                        break
                
                try:
                    tool_name = tool["name"]
                    keys = list(tool.keys())
                    tool_args = tool[keys[1]]
                except Exception as e:
                    messages += [HumanMessage(content=f"{tool}工具调用格式错误:{e}")]
                    break
                tool_result = tools[tool_name].invoke(tool_args)

                web_research_result.append(str(tool_result))
                messages += [HumanMessage(content=f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")]
            
        else:
            break
    
    return {
        "search_query": [state["search_query"]],
        "web_research_result": web_research_result,
    }

def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """
    LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config=config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1

    current_data = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_data=current_data,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"])
    )

    llm = ChatOpenAI(
        model=configurable.reflection_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=configurable.reflection_base_url
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.Knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"])
    }

def evaluate_research(state: ReflectionState, config: RunnableConfig) -> OverallState:
    """
    LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config=config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )

    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx)
                }
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]

def finalize_answer(state: OverallState, config:RunnableConfig):
    """
    LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config=config)

    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["message"]),
        summaries="\n---\n\n".join(state["web_research_result"])
    )

    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=configurable.answer_base_url,
        model=configurable.answer_model,
        temperature=0,
        max_retries=2
    )

    result = llm.invoke(formatted_prompt)
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)
    
    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }

builder = StateGraph(OverallState, context_schema=Configuration)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "generate_query")

builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")

builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)

builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")