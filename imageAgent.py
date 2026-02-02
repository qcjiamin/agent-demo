from typing import TypedDict, List, Annotated
from operator import add
from langchain_openai import ChatOpenAI
from custom.request import generate_image_by_text
from langchain_core.messages import HumanMessage
import json
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    api_key=os.getenv("DASHSCOPE_API_KEY", None),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

vlllm = ChatOpenAI(
    model="qwen3-vl-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY", None),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class AgentState(TypedDict):
    # 用户原始输入
    user_input: str
    # 当前的提示词方案
    current_prompt: str
    # 生成的图片 URL 或 Base64
    image_data: str
    # 艺术总监（Reviewer）的反馈意见
    feedback: str
    score: int
    is_passed: bool
    # 迭代次数，防止无限循环
    iteration_count: int
    # 历史记录
    history: Annotated[List[str], add]

# 提示词优化节点
def refiner_node(state: AgentState):
    print("--- 正在优化设计方案 ---")
    # 这里会调用 LLM，根据 state['user_input'] 和 state['feedback'] 生成新 Prompt
    
    optimize_prompt = f"""你是一个图片生成提示词优化专家，请根据用户输入和修改意见，优化当前的图片生成提示词。
    要求优化后的提示词更加注重细节，让图片具有电影画质
    直接返回优化后的提示词，不要添加任何多余的文字
    用户的提示词: {state['current_prompt'] if state['current_prompt'] else state['user_input']}
    修改意见： {state['feedback']}
    """ 
    new_prompt = llm.invoke(optimize_prompt).content
    print(f"优化后的提示词: {new_prompt}")
    return {"current_prompt": new_prompt, "iteration_count": state['iteration_count'] + 1}

# 绘图执行节点
# todo: 错误处理（重试、添加错误处理节点等）
def generator_node(state: AgentState):
    print(f"--- 正在生成图片 (第 {state['iteration_count']} 次尝试) ---")
    image_urls = generate_image_by_text(state['current_prompt'])

    if image_urls:
        print("图片生成成功，URL列表：")
        for idx, url in enumerate(image_urls, 1):
            print(f"{idx}. {url}")
        image_url = image_urls[0]
        return {"image_data": image_url}
    else:
        print("图片生成失败！")
# 质量评审节点
def reviewer_node(state: AgentState):
    print("--- 艺术总监正在评审 ---")
    reviewer_prompt = f"""你是一位拥有 15 年经验的资深艺术总监。你的任务是根据 图片生成提示词和生成的图片评审 AI 生成的图片质量，给它评分，并给出修改意见。
        评审维度：
            指令契合度 (Alignment)： 画面是否完全遵循了用户的原始需求和 Prompt 中的关键细节？
            视觉质量 (Quality)： 构图是否平衡？是否存在明显的 AI 伪影（如多余的手指、扭曲的物体）？
            审美水平 (Aesthetic)： 色彩搭配是否高级？光影处理是否符合物理逻辑？
        要求你用以下json格式返回评审结果, 不返回任何其他内容，只返回json数据：
        {{
            "score": 0-100,
            "is_passed": true/false,
            "feedback": "给设计师的具体修改建议（如果不通过，请务必提供明确的参数调整或提示词优化建议）"
        }}
        这是提示词：{state['current_prompt']}
        图片为当前输入的图片地址
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": reviewer_prompt},
            {
                "type": "image_url",
                "image_url": {"url": state['image_data']} # image_data 是图片的 URL
            },
        ]
    )
    
    response = vlllm.invoke([message])
    review_result = response.content
    print(f"评审结果: {review_result}")
    return json.loads(review_result)

    # vlllm.invoke(reviewer_prompt, )

    # # 模拟评审逻辑：如果次数小于2，就故意给点修改意见
    # if state['iteration_count'] < 2:
    #     return {"feedback": "构图稍显拥挤，请增加留白。"}
    # else:
    #     return {"feedback": "PASS"}

# 构建图
# 1. 初始化图
workflow = StateGraph(AgentState)

# 2. 添加节点
workflow.add_node("refiner", refiner_node)
workflow.add_node("generator", generator_node)
workflow.add_node("reviewer", reviewer_node)

# 3. 设置入口
workflow.set_entry_point("refiner")

# 4. 设置边（连线）
workflow.add_edge("refiner", "generator")
workflow.add_edge("generator", "reviewer")

# 5. 设置条件边（决策循环）
def should_continue(state: AgentState):
    # 如果通过评审，直接结束
    if state["is_passed"] == True:
        return "end"
    # 如果已经迭代2次还未通过，也结束流程
    if state["iteration_count"] >= 2:
        print(f"已达到最大迭代次数 ({state['iteration_count']} 次)，返回当前结果")
        return "end"
    return "continue"

workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {
        "continue": "refiner", # 回滚重修
        "end": END            # 交付任务
    }
)

# 6. 编译
app = workflow.compile()


if __name__ == "__main__":
    # 初始化状态
    initial_state = {
        "user_input": "一只可爱的橘猫坐在窗台上，阳光洒在它身上",
        "current_prompt": "",
        "image_data": "",
        "feedback": "",
        "score": 0,
        "is_passed": False,
        "iteration_count": 0,
        "history": []
    }
    
    # 运行智能体
    result = app.invoke(initial_state)
    
    print("\n=== 最终结果 ===")
    print(f"迭代次数: {result['iteration_count']}")
    print(f"最终提示词: {result['current_prompt']}")
    print(f"图片地址: {result['image_data']}")
    print(f"评分: {result['score']}")
    print(f"反馈: {result['feedback']}")


#     from langgraph.checkpoint.memory import MemorySaver

# # 1. 初始化内存存储（开发环境用内存，生产环境建议用 Sqlite/Postgres）
# memory = MemorySaver()

# # 2. 编译图时挂载记忆
# app = workflow.compile(checkpointer=memory)

# # 3. 运行任务时提供 thread_id
# config = {"configurable": {"thread_id": "user_001"}}
# app.invoke(input_data, config=config)