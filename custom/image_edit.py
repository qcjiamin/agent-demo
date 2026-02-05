import requests
import os
from dotenv import load_dotenv
load_dotenv()

# -------------------------- 全局配置（只需配置一次） --------------------------
# 方式1：测试用，直接赋值API密钥（生产环境禁用硬编码！）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", None)
# 方式2：生产用，读取系统环境变量（推荐，避免密钥泄露）
# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 生图接口固定地址
GENERATE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


# -------------------------- 封装后的生图方法（仅参数为text） --------------------------
def image_style_change(style, image_url):
    # 构造请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
    }
    # 构造请求体，核心：将参数text传入提示词位置
    payload = {
        "model": "qwen-image-edit-max-2026-01-16",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_url},
                        {"text": f"将图片转换为{style}风格"}  # 封装的核心：使用方法参数style
                    ]
                }
            ]
        },
        "parameters": {
            "n": 1,
            "negative_prompt": "",
            "watermark": False
        }
    }

    try:
        # 发送POST请求
        response = requests.post(
            url=GENERATE_URL,
            headers=headers,
            json=payload,
            timeout=60  # 生图接口耗时较长，保留60秒超时
        )
        response.raise_for_status()  # 捕获HTTP状态码错误（如401/403/500）
        result = response.json()

        # 提取图片链接并返回
        # 实际返回结构：output.choices[0].message.content[{image: url, type: image}]
        if "output" in result and "choices" in result["output"]:
            choices = result["output"]["choices"]
            if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                content = choices[0]["message"]["content"]
                # 提取所有 type 为 image 的项的 image 字段
                image_urls = [item["image"] for item in content if item.get("type") == "image"]
                if image_urls:
                    return image_urls
        
        print(f"生图失败：接口未返回图片数据，响应内容：{result}")
        return None

    # 捕获所有网络请求相关错误（超时、连接失败、DNS错误等）
    except requests.exceptions.RequestException as e:
        print(f"网络请求异常：{e}")
        return None
    # 捕获其他未知错误
    except Exception as e:
        resp_text = response.text if 'response' in locals() else '无响应数据'
        print(f"程序执行异常：{e}，接口原始响应：{resp_text}")
        return None
    
def generate_final(prompt, person_url, hourse_url):
    # 构造请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
    }
    # 构造请求体，核心：将参数text传入提示词位置
    payload = {
        "model": "qwen-image-edit-max-2026-01-16",
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": person_url},
                        {"image": hourse_url},
                        {"text": prompt}
                    ]
                }
            ]
        },
        "parameters": {
            "n": 1,
            "negative_prompt": "",
            "watermark": False
        }
    }

    try:
        # 发送POST请求
        response = requests.post(
            url=GENERATE_URL,
            headers=headers,
            json=payload,
            timeout=60  # 生图接口耗时较长，保留60秒超时
        )
        response.raise_for_status()  # 捕获HTTP状态码错误（如401/403/500）
        result = response.json()

        # 提取图片链接并返回
        # 实际返回结构：output.choices[0].message.content[{image: url, type: image}]
        if "output" in result and "choices" in result["output"]:
            choices = result["output"]["choices"]
            if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                content = choices[0]["message"]["content"]
                # 提取所有 type 为 image 的项的 image 字段
                image_urls = [item["image"] for item in content if item.get("type") == "image"]
                if image_urls:
                    return image_urls
        
        print(f"生图失败：接口未返回图片数据，响应内容：{result}")
        return None

    # 捕获所有网络请求相关错误（超时、连接失败、DNS错误等）
    except requests.exceptions.RequestException as e:
        print(f"网络请求异常：{e}")
        return None
    # 捕获其他未知错误
    except Exception as e:
        resp_text = response.text if 'response' in locals() else '无响应数据'
        print(f"程序执行异常：{e}，接口原始响应：{resp_text}")
        return None
