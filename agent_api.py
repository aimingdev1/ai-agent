from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openai import OpenAI
from ddgs import DDGS
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime

# 1. 创建FastAPI应用
app = FastAPI()

# 2. 配置DeepSeek客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)



# 3. 三个工具函数
def calculator_tool(expression):
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except:
        return "计算失败,请输入正确算式,比如 2+3"


def search_tool(query):
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "未找到相关结果"
        answer = "搜索结果如下\n"
        for i, r in enumerate(results, 1):
            answer += f"{i}. {r['title']}\n{r['href']}\n{r['body'][:100]}...\n"
        return answer
    except:
        return "搜索失败,请稍后再试"


# 4. Agent核心大脑
def agent_think(user_input):
    messages = [
        {"role": "system", "content": """你是一个智能助手，可以调用两个工具：
1. 计算器：需要计算时，回复必须以 CALCULATE: 开头
2. 搜索：需要查信息时，回复必须以 SEARCH: 开头
其他情况直接回复。"""},
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
    )

    ai_text = response.choices[0].message.content
    return ai_text if ai_text else ""


# 5. API接口
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Agent is running"}
@app.get("/chat")
def chat_endpoint(msg: str = ""):
    if not msg:
        return {"reply": "你好！我是你的AI助手，可以帮你计算、搜索，请随便问我吧！"}

    # 调用Agent大脑
    ai_response = agent_think(msg)

    # 判断调用哪个工具
    if ai_response.startswith("CALCULATE:"):
        expression = ai_response.replace("CALCULATE:", "").strip()
        final = calculator_tool(expression)
    elif ai_response.startswith("SEARCH:"):
        query = ai_response.replace("SEARCH:", "").strip()
        final = search_tool(query)
    else:
        final = ai_response

    return {"reply": final}


# 6. 一个简单的网页首页
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Agent</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #f5f5f5;
                display: flex;
                justify-content: center;
                padding: 40px 20px;
                min-height: 100vh;
            }
            .container {
                width: 100%;
                max-width: 700px;
                background: white;
                border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                overflow: hidden;
                display: flex;
                flex-direction: column;
                height: 85vh;
                max-height: 800px;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px 24px;
                font-size: 20px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }
            .chat-area {
                flex: 1;
                overflow-y: auto;
                padding: 24px;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .message {
                display: flex;
                gap: 10px;
                animation: fadeIn 0.3s ease;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(8px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .message.user { justify-content: flex-end; }
            .bubble {
                max-width: 80%;
                padding: 12px 16px;
                border-radius: 12px;
                line-height: 1.6;
                font-size: 15px;
                white-space: pre-wrap;
            }
            .message.user .bubble {
                background: #667eea;
                color: white;
                border-bottom-right-radius: 4px;
            }
            .message.ai .bubble {
                background: #f0f0f5;
                color: #333;
                border-bottom-left-radius: 4px;
            }
            .input-area {
                display: flex;
                padding: 16px 24px;
                border-top: 1px solid #eee;
                gap: 12px;
                background: #fafafa;
            }
            .input-area input {
                flex: 1;
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                font-size: 15px;
                outline: none;
                transition: border 0.2s;
            }
            .input-area input:focus {
                border-color: #667eea;
            }
            .input-area button {
                padding: 12px 24px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: 500;
                cursor: pointer;
                transition: background 0.2s;
            }
            .input-area button:hover {
                background: #5a6fd6;
            }
            .loading {
                display: flex;
                gap: 6px;
                padding: 12px 16px;
            }
            .loading span {
                width: 8px;
                height: 8px;
                background: #999;
                border-radius: 50%;
                animation: bounce 1.4s infinite ease-in-out both;
            }
            .loading span:nth-child(1) { animation-delay: -0.32s; }
            .loading span:nth-child(2) { animation-delay: -0.16s; }
            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">🤖 我的AI助手</div>
            <div class="chat-area" id="chat"></div>
            <div class="input-area">
                <input id="msg" type="text" placeholder="问我任何问题..." onkeypress="if(event.key==='Enter')send()">
                <button onclick="send()">发送</button>
            </div>
        </div>
        <script>
            function addMessage(role, text) {
                const chat = document.getElementById('chat');
                const div = document.createElement('div');
                div.className = 'message ' + role;
                const bubble = document.createElement('div');
                bubble.className = 'bubble';
                bubble.textContent = text;
                div.appendChild(bubble);
                chat.appendChild(div);
                chat.scrollTop = chat.scrollHeight;
                return bubble;
            }

            async function send() {
                const input = document.getElementById('msg');
                const msg = input.value.trim();
                if (!msg) return;

                addMessage('user', msg);
                input.value = '';

                const loadingBubble = addMessage('ai', '...');

                try {
                    const resp = await fetch('/chat?msg=' + encodeURIComponent(msg));
                    const data = await resp.json();
                    loadingBubble.textContent = data.reply;
                } catch(e) {
                    loadingBubble.textContent = '出错了，请稍后再试';
                }
            }
        </script>
    </body>
    </html>
    """