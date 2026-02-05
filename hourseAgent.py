# 照片和与马的交互提示词
# 生成风格->模型随机返回3个，固定喜庆类1个
# 照片去背景？
# 照片转风格
# 生成该风格的马
# 模型优化交互提示词，如果没有交互，那么生成交互提示
# 生图
from langgraph.graph import StateGraph, START, END, MessagesState
from typing_extensions import TypedDict, Annotated
import os
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.messages import AnyMessage, HumanMessage
import operator
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from custom.request import generate_image_by_text
from custom.image_edit import image_style_change, generate_final
load_dotenv()

system_prompt = """你是一个优秀的艺术设计专家, 同时也擅长提示词工程。"""

llm = ChatOpenAI(
    model="qwen3-max-2026-01-23",
    api_key=os.getenv("DASHSCOPE_API_KEY", None),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    system_prompt=system_prompt
)

class ImageStyles(BaseModel):
    # 限制列表长度并添加描述
    styles: List[str] = Field(
        description="包含4种独特的图片风格名称的列表",
        min_items=4,
        max_items=4
    )
structured_llm = llm.with_structured_output(ImageStyles)
took_llm = llm.bind_tools([])

# Define the state
class MessageState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    person: str
    person_with_style: str
    hourse_with_style: str
    final_image: str
    styles: List[str]
    style: str

# Define nodes
def style_generate(state: MessageState):
    """生成随机的风格名称"""
    structured_llm.invoke({"messages": [
        HumanMessage(content="请生成4种随机的图片风格名称, 风格名字不要重复")
    ]})
    return {"styles": structured_llm(state["messages"])}

def hourse_generate(state: MessageState):
    """生成带有指定风格的马的图片"""
    prompt = f"请生成一张符合以下风格的马的图片: {state['style']}"
    image_urls = generate_image_by_text(prompt)
    if image_urls:
        print("马的图片生成成功，URL列表：")
        for idx, url in enumerate(image_urls, 1):
            print(f"{idx}. {url}")
        image_url = image_urls[0]
        return {"hourse_with_style": image_url}
    else:
        print("图片生成失败！")

def person_generate(state: MessageState):
    """转换照片中的人物为指定风格"""
    image_url = state['person']  # Use the person image URL from the state

    image_urls = image_style_change(state['style'], image_url)
    if image_urls:
        print("人物图片转换成功，URL列表：")
        for idx, url in enumerate(image_urls, 1):
            print(f"{idx}. {url}")
        return {"person_with_style": image_urls[0]}
    else:
        print("人物图片转换失败！")

def image_generate(state: MessageState):
    """生成图片"""
    image_urls = generate_final(f"图1是人物，图二是马，绘制一张人骑着马在草原上驰骋的图片，风格为{state['style']}", state['person_with_style'], state['hourse_with_style'])
    if image_urls:
        print("图片生成成功，URL列表：")
        for idx, url in enumerate(image_urls, 1):
            print(f"{idx}. {url}")
        return {"final_image": image_urls[0]}
    else:
        print("图片生成失败！")


agent_builder = StateGraph(MessageState)
agent_builder.add_node("style_generate", style_generate)
agent_builder.add_node("hourse_generate", hourse_generate)
agent_builder.add_node("person_generate", person_generate)
agent_builder.add_node("image_generate", image_generate)




# Compile the agent
agent = agent_builder.compile()
# Invoke the agent
messages = [HumanMessage(content=[
    {type: "text", "text": ""},
    {type: "image_url", "image_url": "https://example.com/horse.jpg"}
])]
messages = agent.invoke({"messages": messages})