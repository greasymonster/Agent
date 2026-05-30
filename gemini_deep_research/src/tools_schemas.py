from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List

import requests

embedding_model = HuggingFaceBgeEmbeddings(
    model_name="bge-large-zh-v1.5",
    model_kwargs={"devices": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True}
)

db = FAISS.load_local("faiss", embedding_model, allow_dangerous_deserialization=True)

@tool
def knowledge_search(query: str) -> str:
    """
    根据query进行知识库查询，获取相关知识
    """
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1})
    results = retriever.get_relevant_documents(query=query)
    print(results)
    return results

@tool
def web_search(query: str) -> str:
    """
    根据query进行互联网搜索，获取相关知识
    """
    links=[]
    print(f"Searching for: {query}")
    response = requests.get(f"http://localhost:8088/search?format=json&q={query}&language=zh-CN&time_range=&safesearch=0&categories=general")
    print(len(response.json()["results"]))
    results = [result for result in response.json()["results"] if result["engine"] == "baidu"]
    print(len(results))
    results = results[:20]
    for result in results:
        link = results["url"]
        if link not in links:
            links.append(link)

    all_text = []
    for link in links:
        text = fetch_webpage_text(link) 
        if non_chinese_ratio(text) > 0.5:
            continue
        print("++++++++++++++++++++++++++++++++++++++++++")
        print(text)
        print("++++++++++++++++++++++++++++++++++++++++++")
        if text:
            all_text.append({"url": link, "text": text})
    
    print(all_text)
    return all_text

def non_chinese_ratio(text):
    """
    计算文本中非中文字符的比例
    :param text: 输入文本字符串
    :return: 非中文字符的比例（0.0~1.0）
    """
    if not text.strip():
        return 1.0
    
    non_chinese_count = 0
    for char in text:
        if not '\u4e00' <= char <='\u9fa5':
            non_chinese_count += 1
    
    return non_chinese_count / len(text)

def fetch_webpage_text(url):
    JINA_BASE_URL = "https://r.jina.ai/"
    full_url = f"{JINA_BASE_URL}{url}"

    try:
        resp = requests.get(full_url)
        if resp.status_code == 200:
            return resp.text
        else:
            text = resp.text
            return ""
    except Exception as e:
        return ""

class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )

class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )