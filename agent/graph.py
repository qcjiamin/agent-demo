from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    api_key=os.getenv("DASHSCOPE_API_KEY", None),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
LIMIT = 500


class ExpenseState(TypedDict):
    invoice_text: str
    amount: float
    approved: bool
    waiting_human: bool


def parse_invoice_node(state: ExpenseState):
    prompt = f"提取总金额，只返回数字：{state['invoice_text']}"
    amount = float(llm.invoke(prompt).content)
    return {"amount": amount}


def policy_check_node(state: ExpenseState):
    if state["amount"] <= LIMIT:
        return {"approved": True}
    return {"waiting_human": True}


# ❗ 不再 input，而是“暂停”
def human_review_node(state: ExpenseState):
    return {"waiting_human": True}


def bookkeeping_node(state: ExpenseState):
    return {}


graph = StateGraph(ExpenseState)
graph.add_node("parse_invoice", parse_invoice_node)
graph.add_node("policy_check", policy_check_node)
graph.add_node("human_review", human_review_node)
graph.add_node("bookkeeping", bookkeeping_node)

graph.set_entry_point("parse_invoice")
graph.add_edge("parse_invoice", "policy_check")


def route(state):
    if state.get("approved"):
        return "bookkeeping"
    return "human_review"


graph.add_conditional_edges("policy_check", route)
graph.add_edge("human_review", END)

app_graph = graph.compile()
