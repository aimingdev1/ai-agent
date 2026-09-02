from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openai import OpenAI
from ddgs import DDGS
import os
import re
import math
import time          # ⏱ 计算每次请求耗时（工程化：可观测）
import logging        # 📝 记录日志（工程化：可观测）
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

# ===== 工程化模块一：日志（每次调用留痕，出问题能回查"哪一步慢、哪一步错"） =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),                              # 同步打到控制台
        logging.FileHandler("agent.log", encoding="utf-8"),    # 同时存到文件
    ],
)
logger = logging.getLogger("agent")

# ===== 工程化模块二：结果缓存（同一个问题再问，直接返回上次答案，0成本0延迟） =====
_CACHE = {}          # {问题: 上次的回答}
_CACHE_MAX = 100     # 最多缓存100条，防止内存无限膨胀

# ===== 工程化模块三：简易统计（用来算缓存命中率、看工具被调多少次） =====
_stats = {
    "total": 0,
    "cache_hit": 0,
    "tool": {"CALCULATE": 0, "SEARCH": 0, "RAG": 0, "DIRECT": 0},
}



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


# ===== RAG 知识库检索工具 =====
RAG_DOCS = [
    "RAG全称是检索增强生成，核心是先检索再生成，用来解决大模型知识有截止日期、答不了私有文档的问题。",
    "ReAct是一种智能体推理框架，让模型交替进行思考和行动，可以调用外部工具，并支持多步推理。",
    "向量数据库用来存储文本的向量表示，支持按相似度快速找出最相关的片段。",
    "Embedding是把文字变成一串数字，语义相近的文字对应的向量也相近，是检索的基础。",
]

# 1) 分块
def _rag_split(text, max_len=100):
    if len(text) <= max_len:
        return [text]
    return [text[i:i + max_len] for i in range(0, len(text), max_len)]

_RAG_CHUNKS = []
for _d in RAG_DOCS:
    _RAG_CHUNKS.extend(_rag_split(_d))

# 2) 向量化（TF-IDF，替代真实 Embedding，原理等价：文字→数字向量→算相似度）
def _rag_tokenize(text):
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())

_RAG_VOCAB = {}
for _c in _RAG_CHUNKS:
    for _w in set(_rag_tokenize(_c)):
        if _w not in _RAG_VOCAB:
            _RAG_VOCAB[_w] = len(_RAG_VOCAB)
_RAG_VOCAB_SIZE = len(_RAG_VOCAB)

_RAG_N = len(_RAG_CHUNKS)
_rag_df = {}
for _c in _RAG_CHUNKS:
    for _w in set(_rag_tokenize(_c)):
        _rag_df[_w] = _rag_df.get(_w, 0) + 1
_RAG_IDF = {_w: math.log((_RAG_N + 1) / (_rag_df[_w] + 1)) + 1 for _w in _rag_df}

def _rag_vector(text):
    vec = [0.0] * _RAG_VOCAB_SIZE
    words = _rag_tokenize(text)
    if not words:
        return vec
    tf = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    for w, c in tf.items():
        if w in _RAG_VOCAB:
            vec[_RAG_VOCAB[w]] = (c / len(words)) * _RAG_IDF.get(w, 1.0)
    return vec

_RAG_CHUNK_VECTORS = [_rag_vector(c) for c in _RAG_CHUNKS]

def _rag_cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# 3) RAG 检索工具
def rag_tool(query):
    """知识库检索工具：智能体用它查私有知识库"""
    q_vec = _rag_vector(query)
    scored = [(_rag_cos(q_vec, cv), i) for i, cv in enumerate(_RAG_CHUNK_VECTORS)]
    scored.sort(reverse=True)
    top = scored[:3]
    result = "\n".join(f"[{j+1}] {_RAG_CHUNKS[i]}" for j, (_, i) in enumerate(top))
    return f"检索到的相关资料:\n{result}"

# 4) RAG 作答
def rag_answer(question, knowledge):
    prompt = f"""你是问答助手。请只根据下面【资料】回答问题，如果资料里没有答案，就如实回答"资料中没有相关信息"，不要编造。

【资料】
{knowledge}

【问题】{question}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return response.choices[0].message.content


# 4. Agent核心大脑
def agent_think(user_input):
    messages = [
        {"role": "system", "content": """你是智能助手的路由模块，必须判断用户的问题该用哪个工具处理。严格按规则选择：

1. 涉及数字计算、算式求解 → 必须回复 CALCULATE: 算式（例如 CALCULATE: 12*34）
2. 涉及实时/最新信息：新闻、天气、行情、最新型号、近期动态 → 必须回复 SEARCH: 检索词
3. 涉及知识库主题：RAG、检索增强生成、ReAct、智能体、向量数据库、Embedding → 必须回复 RAG: 检索词
4. 仅当问题是打招呼、闲聊、创作类等与以上工具都无关时，才直接回答

示例：
用户：帮我算 3乘以8 → CALCULATE: 3*8
用户：今天有什么AI新闻 → SEARCH: 今日AI新闻
用户：什么是Embedding → RAG: Embedding
用户：写一句鼓励我的话 → （直接回答，不加前缀）"""},
        {"role": "user", "content": user_input}
    ]

    t0 = time.time()
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
    )
    # 记录这一步的耗时和token消耗（定位问题时能分清是"决策慢"还是"工具慢"）
    tokens = getattr(response.usage, "total_tokens", 0)
    logger.info("LLM决策 | 耗时%.2fs | tokens=%s", time.time() - t0, tokens)

    ai_text = response.choices[0].message.content
    return ai_text if ai_text else ""


# 5. API接口
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Agent is running"}

@app.get("/stats")
def stats_endpoint():
    """可观测性：缓存命中率 + 各工具调用次数，验证缓存真的在工作"""
    total = _stats["total"]
    hits = _stats["cache_hit"]
    return {
        "total": total,
        "cache_hit": hits,
        "cache_miss": total - hits,
        "hit_rate": f"{hits / total * 100:.1f}%" if total else "N/A",
        "cache_size": len(_CACHE),
        "tool_usage": _stats["tool"],
    }

@app.get("/chat")
def chat_endpoint(msg: str = ""):
    if not msg:
        return {"reply": "你好！我是你的AI助手，可以帮你计算、搜索、查知识库，请随便问我吧！"}

    _stats["total"] += 1
    key = msg.strip()

    # ① 先查缓存：命中直接返回上次答案（不调LLM、不花钱、毫秒级）
    if key in _CACHE:
        _stats["cache_hit"] += 1
        logger.info("缓存命中 | 问题=%s...", key[:30])
        reply, route = _CACHE[key]
        return {"reply": reply, "cached": True, "route": route}

    # ② 缓存未命中，走正常流程
    t0 = time.time()
    ai_response = agent_think(msg)

    # 判断调用哪个工具
    if ai_response.startswith("CALCULATE:"):
        route = "CALCULATE"
        expression = ai_response.replace("CALCULATE:", "").strip()
        final = calculator_tool(expression)
    elif ai_response.startswith("SEARCH:"):
        route = "SEARCH"
        query = ai_response.replace("SEARCH:", "").strip()
        final = search_tool(query)
    elif ai_response.startswith("RAG:"):
        route = "RAG"
        query = ai_response.replace("RAG:", "").strip()
        knowledge = rag_tool(query)       # ① 检索私有知识库拿资料
        final = rag_answer(msg, knowledge)  # ② 把资料回灌给 LLM 生成答案
    else:
        route = "DIRECT"
        final = ai_response

    # ③ 记录这一单：走了哪条路、总共花多久
    _stats["tool"][route] += 1
    logger.info("请求完成 | 路由=%s | 总耗时%.2fs | 问题=%s...", route, time.time() - t0, key[:30])

    # ④ 存入缓存（满了就挤掉最早的一条，简单FIFO；连同路由一起存，命中时也能返回route）
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (final, route)

    return {"reply": final, "cached": False, "route": route}


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
