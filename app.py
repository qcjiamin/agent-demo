from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from agent.graph import app_graph

app = FastAPI()
tasks = {}  # 实际应使用数据库


class InvoiceIn(BaseModel):
    text: str


@app.post("/submit")
def submit_invoice(data: InvoiceIn):
    task_id = str(uuid.uuid4())

    state = {
        "invoice_text": data.text
    }

    result = app_graph.invoke(state)
    tasks[task_id] = result

    if result.get("waiting_human"):
        return {"task_id": task_id, "status": "need_approval", "amount": result["amount"]}

    return {"task_id": task_id, "status": "done", "approved": True}


class ApprovalIn(BaseModel):
    task_id: str
    approved: bool

@app.post("/approve")
def approve(data: ApprovalIn):
    state = tasks[data.task_id]
    state["approved"] = data.approved
    state["waiting_human"] = False

    result = app_graph.invoke(state)
    tasks[data.task_id] = result

    return {"status": "finished", "approved": data.approved}