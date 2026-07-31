import random
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

#计算工具
def calculator_tool(expression):
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except:
        return "计算失败,请输入正确算式,比如 2+3"

def search_tool(query):
    try:
        from ddgs import DDGS
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "未找到相关结果"
        answer = "搜索结果如下\n"
        for i,r in enumerate(results,1):
            answer += f"{i}. {r['title']}\n {r['href']}\n {r['body'][:100]}...\n"
        return answer
    except Exception:
        return "搜索失败,请稍后再试"

"""
#AI大脑
def ai_brain(user_input):
    responses = ["我听到了,你说的意思是: " + user_input,
                 "我不太明白你的意思",
                 "这是个好问题,让我想想"
                 ]
    return random.choice(responses)


# 升级版AI大脑:判断是不是数学题
def ai_brain_v2(user_input):
    #检查输入是否有数字
    if any(char.isdigit() for char in user_input):
        return calculator_tool(user_input)
    else:
        return   ai_brain(user_input)
"""

# 真正的大模型大脑
def ai_brain_v3(user_input):
    messages = [
        {"role": "system", "content": """你是一个智能助手，可以调用两个工具：

    1. 计算器：如果用户让你做数学计算，回复必须以 CALCULATE: 开头，后面只写算式。例如：CALCULATE: 3*15

    2. 搜索：如果用户问实时信息、百科知识、新闻、人物、事件等你不知道或需要查证的内容，回复必须以 SEARCH: 开头，后面只写搜索关键词。例如：SEARCH: 2024年诺贝尔物理学奖得主

    如果是普通问候或闲聊，直接回复即可。
    """},
        {"role": "user", "content": user_input}
    ]
    # 2. 给DeepSeek打电话，把指令和用户的话发过去
    response = client.chat.completions.create(
        model = "deepseek-v4-flash",
        messages = messages,
        stream = False,
    )
    # 3. 从DeepSeek的回信中，取出AI说的话
    ai_text = response.choices[0].message.content
    return ai_text

while True:
    user_message = input("你: ")
    if user_message == "拜拜":
        print("AI: 再见! ")
        break
    ai_response = ai_brain_v3(user_message)
 #检查AI是不是在“请求调用计算器”
    if ai_response and ai_response.startswith("CALCULATE:"):
        expression = ai_response.replace("CALCULATE:", "").strip()
        # 亲手调用我们之前写的计算器工具，算出真实结果
        final_result = calculator_tool(expression)
        print("AI：" + final_result)
    elif ai_response and ai_response.startswith("SEARCH:"):
        query = ai_response.replace("SEARCH:", "").strip()
        final_result = search_tool(query)
        print("AI：" + final_result)
    else:
        print("AI: " + ai_response)


