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


# 🧠 工作流状态
class ExpenseState(TypedDict):
    invoice_text: str
    amount: float
    approved: bool


# 🧾 节点1：解析发票
def parse_invoice_node(state: ExpenseState):
    prompt = f"从下面发票文本中提取总金额，只返回数字：\n{state['invoice_text']}"
    result = llm.invoke(prompt).content
    return {"amount": float(result)}


# 📏 节点2：合规判断
def policy_check_node(state: ExpenseState):
    if state["amount"] <= LIMIT:
        return {"approved": True}
    return {"approved": False}


# 🤝 节点3：人工审批（可暂停）
def human_review_node(state: ExpenseState):
    print(f"⚠️ 金额 {state['amount']} 超标，等待财务审批")
    approval = input("是否批准？(y/n): ")
    return {"approved": approval == "y"}


# 💰 节点4：入账执行
def bookkeeping_node(state: ExpenseState):
    if state["approved"]:
        print("✅ 报销完成")
    else:
        print("❌ 报销被拒")
    return {}


# 🏗 构建图
graph = StateGraph(ExpenseState)

graph.add_node("parse_invoice", parse_invoice_node)
graph.add_node("policy_check", policy_check_node)
graph.add_node("human_review", human_review_node)
graph.add_node("bookkeeping", bookkeeping_node)

graph.set_entry_point("parse_invoice")
graph.add_edge("parse_invoice", "policy_check")


# 条件路由
def route(state: ExpenseState):
    return "bookkeeping" if state["approved"] else "human_review"


graph.add_conditional_edges("policy_check", route)
graph.add_edge("human_review", "bookkeeping")
graph.add_edge("bookkeeping", END)

app = graph.compile()


# ▶️ 运行
initial_state = {"invoice_text": "酒店住宿费用，总计 860 元"}
app.invoke(initial_state)
