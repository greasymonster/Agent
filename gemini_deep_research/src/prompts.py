from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""

query_writer_instructions = """你的目标是生成高质量且多样化的搜索查询。这些查询将用于先进的自动化检索工具，该工具能够分析复杂结果、追踪链接并整合信息。

操作指南：
- 原则上使用单条搜索查询，仅当原始问题涉及多个方面且一条查询不足时才添加额外查询
- 每条查询应聚焦原始问题的具体某个方面
- 生成查询不得超过 {number_queries} 条
- 主题宽泛时应生成多于1条的多样化查询
- 避免生成相似查询，1条足够时无需多条

格式要求：
- 响应必须为包含以下两个精确键名的JSON对象：
   - "rationale": 简要说明查询相关性的理由
   - "query": 搜索查询的列表

示例：

主题：去年苹果股票和iPhone购买者数量哪个增长更快
```json
{{
    "rationale": "为准确回答此对比增长问题，需要苹果股票表现和iPhone销售数据的具体指标。这些查询精确指向所需财务信息：公司收入趋势、产品具体销量数据和同期股价变动，以便直接对比。",
    "query": ["苹果2024财年总收入增长", "iPhone 2024财年销量增长", "苹果股票2024财年涨幅"],
}}

上下文：{research_topic}"""


web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""
web_searcher_instructions = """针对"{research_topic}"执行互联网或知识库搜索，收集可靠信息，并整合成可验证的文本成果。

操作指南：
- 执行多次多样化搜索以获取全面信息。
- 整合关键发现时，需严格记录每条信息的具体来源。
- 输出内容应为基于搜索结果的完整摘要或报告。
- 仅包含搜索结果中的信息，严禁编造内容。

研究主题：
{research_topic}
"""

reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

reflection_instructions = """您是一位专业研究助理，正在分析关于“{research_topic}”的摘要。

指令：
- 识别知识缺口或需要深入探索的领域，并生成后续查询（1个或多个）
- 若提供的摘要足以回答用户问题，则无需生成后续查询
- 若存在知识缺口，生成有助于扩展理解的后续查询
- 重点关注未充分涵盖的技术细节、具体实现方法或新兴趋势

要求：
- 确保后续查询是自包含的，并包含网络搜索所需的必要上下文

输出格式：
- 严格按以下键名生成JSON对象：
   - "is_sufficient": true 或 false
   - "knowledge_gap": 描述缺失或需要澄清的信息
   - "follow_up_queries": 提出解决此缺口的具体问题

示例：
```json
{{
    "is_sufficient": true, // 或 false
    "knowledge_gap": "摘要缺乏性能指标和基准测试信息", // 若is_sufficient为true则留空
    "follow_up_queries": ["评估[特定技术]的典型性能基准和指标有哪些？"] // 若is_sufficient为true则留空
}}

请仔细分析摘要以识别知识缺口并生成后续查询，然后严格按此JSON格式输出：

摘要内容：
{summaries}
"""

answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step. 
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- Include the sources you used from the Summaries in the answer correctly, use markdown format (e.g. [apnews](https://vertexaisearch.cloud.google.com/id/1-0)). THIS IS A MUST.

User Context:
- {research_topic}

Summaries:
{summaries}"""

answer_instructions = """根据提供的摘要，生成针对用户问题的高质量回答。

指令：
- 作为多阶段研究流程的最终环节（无需在回答中提及此流程）
- 您已获取前期所有研究信息
- 可随时参考用户原始问题
- 基于摘要内容和用户问题生成高质量答案
- 必须正确标注知识来源或链接
- 生成全面详细的内容, 确保使用获取的研究信息的所有内容
- 输出内容不得少于5000字

用户背景：
- {research_topic}

摘要内容：
{summaries}"""
