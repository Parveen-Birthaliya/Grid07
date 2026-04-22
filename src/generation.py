# src/generation.py

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from typing import List, Dict, Optional
import logging
import json
from uuid import uuid4 

from .models import AgentState, SearchResult
from .config import LLM_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

def mock_search(query: str) -> str:
    """Mock search for testing"""
    search_db = {
        "crypto": "Bitcoin hits record high. Ethereum surges 30%. DeFi TVL exceeds $100B.",
        "ai": "OpenAI releases o3 model. Google Gemini shows improvement. Meta AI models advance.",
        "market": "Fed keeps rates steady. Stock market rallies on positive data. Tech leads gains.",
        "ev": "Tesla announces new model. BYD battery tech breakthrough. EV sales surge."
    }
    
    for keyword, result in search_db.items():
        if keyword in query.lower():
            return result
    
    return "Markets show mixed signals. Analysts debate economic outlook."
def duckduckgo_search_tool(query: str) -> List[str]:
    """Real-time search using DuckDuckGo. Returns top 5 results."""
    results = []
    try:
        from ddgs import DDGS
        with DDGS() as client:
            search_results = client.text(query, max_results=5)
            for r in search_results:
                title = r.get("title", "")
                snippet = r.get("body", "")
                formatted = f"{title} — {snippet}"
                results.append(formatted)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        results.append(f"Search error: {str(e)}")
    
    return results if results else ["No results found."]


def decide_search(state: AgentState) -> dict:
    """Node 1: Bot decides what topic to search"""
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=None  # Will use GROQ_API_KEY from env
    )

    prompt = f"""
    You are a bot with this persona:
    {state.matched_bots[0].persona if state.matched_bots else "neutral"}

    The user posted: {state.query}

    Decide ONE topic you want to respond about.
    Generate a short search query to get context.

    Return ONLY JSON (no markdown, no extra text):
    {{"topic": "...", "search_query": "..."}}
    """
    
    response = llm.invoke(prompt)
    data = json.loads(response.content)

    return {
        "topic": data.get("topic", ""),
        "search_query": data.get("search_query", "")
    }


def web_search(state: AgentState) -> dict:
    """Node 2: Execute search query"""
    search_query = state.search_query or state.query
    
    try:
        results_list = duckduckgo_search_tool(search_query)
        # Convert list to string by joining with separator
        results_text = " | ".join(results_list) if results_list else "No results found"
        
        search_result = SearchResult(
            query=search_query,
            results=results_text,
            source="duckduckgo",
            retry_count=0
        )
        return {"search_results": search_result}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": f"Search failed: {str(e)}"}


def draft_post(state: AgentState) -> dict:
    """Node 3: Generate opinionated post"""
    
    if not state.matched_bots:
        return {"error": "No bot matched"}
    
    bot = state.matched_bots[0]
    context = state.search_results.results if state.search_results else ""
    
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )

    prompt = f"""
    You are bot: {bot.bot_id}
    Persona: {bot.persona}
    
    Topic: {state.topic}
    
    Context from search:
    {context}
    
    Generate a STRONG, OPINIONATED X/Twitter post.
    Maximum 280 characters.
    Stay completely in character.
    
    Return ONLY JSON (no markdown):
    {{"post_content": "...", "bot_id": "{bot.bot_id}"}}
    """

    response = llm.invoke(prompt)
    data = json.loads(response.content)

    return {
        "generated_post": data.get("post_content", ""),
        "topic": state.topic
    }


def build_generation_graph():
    """Build and compile LangGraph pipeline"""
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("decide_search", decide_search)
    builder.add_node("web_search", web_search)
    builder.add_node("draft_post", draft_post)

    # Set flow
    builder.set_entry_point("decide_search")
    builder.add_edge("decide_search", "web_search")
    builder.add_edge("web_search", "draft_post")
    builder.add_edge("draft_post", END)

    # Compile
    graph = builder.compile()
    logger.info("Generation graph compiled successfully")
    
    return graph
        