# ==============================
# AI 报销助手（LangChain LCEL 版）
# ==============================

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
load_dotenv()

# 🔹 1. 初始化 LLM（可替换为 Qwen OpenAI 兼容接口）
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    api_key=os.getenv("DASHSCOPE_API_KEY", None),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 🔹 2. 多轮会话存储（支持多用户）
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 🔹 3. 发票金额解析 Prompt
invoice_prompt = ChatPromptTemplate.from_template(
    "从下面发票文本中提取总金额，只返回数字：\n{text}"
)

# LCEL 链
parse_invoice_chain = invoice_prompt | llm | StrOutputParser()

# 让链支持“记忆”
parse_invoice_chain = RunnableWithMessageHistory(
    parse_invoice_chain,
    get_session_history,
    input_messages_key="text",
)

# 🔹 4. 报销规则
LIMIT = 500

def policy_check(amount: float):
    return amount <= LIMIT

# 🔹 5. 人工审批
def human_review(amount):
    print(f"⚠️ 金额 {amount} 超过报销标准，需要人工审批")
    approval = input("财务是否批准？(y/n): ")
    return approval.lower() == "y"

# 🔹 6. 报销主流程
def expense_assistant(invoice_text: str, session_id="user1"):
    print("\n🧾 用户上传发票...")
    
    # AI 解析金额（带会话记忆）
    amount_str = parse_invoice_chain.invoke(
        {"text": invoice_text},
        config={"configurable": {"session_id": session_id}}
    )

    try:
        amount = float(amount_str.strip())
    except:
        print("❌ AI 金额解析失败")
        return "报销失败"

    print(f"🤖 AI 识别金额：{amount}")

    # 规则判断
    if policy_check(amount):
        print("✅ 金额合规，自动报销")
        return "报销成功"

    # 人工审批
    approved = human_review(amount)

    if approved:
        print("✅ 审批通过，报销完成")
        return "审批后报销成功"

    print("❌ 审批拒绝，报销失败")
    return "报销失败"

# ==============================
# ▶️ 模拟运行
# ==============================

if __name__ == "__main__":
    while True:
        text = input("\n请输入发票内容（q退出）：")
        if text == "q":
            break
        result = expense_assistant(text)
        print("📄 结果：", result)
