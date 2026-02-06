from fastapi import FastAPI
from pydantic import BaseModel
import uuid
from hourseAgent import agent as app_graph
from langchain.messages import AnyMessage, HumanMessage
from langgraph.types import Command

app = FastAPI()
tasks = {}  # 实际应使用数据库

class SubmitRequest(BaseModel):
    prompt: str  # 用户上传的照片 URL

class StyleSelectRequest(BaseModel):
    task_id: str
    selected_style: str  # 用户选择的风格
@app.post("/submit")
def submit_task(data: SubmitRequest):
    """提交任务，生成风格列表"""
    # task_id = str(uuid.uuid4())
    
    # !使用 thread_id 调用 agent, 为了后续恢复执行
    config = {"configurable": {"thread_id": 1}}
    
    result = app_graph.invoke(
        {
            "messages": [HumanMessage(content=data.prompt)],
            # "person": data.person_image_url
        },
        config=config
    )
    
    # 检查是否被 interrupt 暂停
    state_snapshot = app_graph.get_state(config)
    
    if state_snapshot.next:  # 如果有下一个节点，说明被暂停了
        # 从 interrupt 值中获取风格列表
        interrupt_value = state_snapshot.tasks[0].interrupts[0].value
        return {
            "task_id": 1,
            "status": "waiting_style_selection",
            "styles": interrupt_value["styles"],
            "message": interrupt_value["message"]
        }
    
    # 如果没有暂停，直接返回最终结果
    return {
        "task_id": 1,
        "status": "completed",
        "final_image": result["final_image"]
    }
@app.post("/select-style")
def select_style(data: StyleSelectRequest):
    """用户选择风格后，恢复执行"""
    config = {"configurable": {"thread_id": data.task_id}}
    
    # 使用 Command 恢复执行，并传入用户选择
    result = app_graph.invoke(
        Command(resume=data.selected_style),
        config=config
    )
    
    return {
        "task_id": data.task_id,
        "status": "completed",
        "final_image": result["final_image"]
    }


# class ApprovalIn(BaseModel):
#     task_id: str
#     approved: bool

# @app.post("/approve")
# def approve(data: ApprovalIn):
#     state = tasks[data.task_id]
#     state["approved"] = data.approved
#     state["waiting_human"] = False

#     result = app_graph.invoke(state)
#     tasks[data.task_id] = result

#     return {"status": "finished", "approved": data.approved}