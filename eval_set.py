# -*- coding: utf-8 -*-
"""
路由准确率评测集
=================
作用：拿 20 道标准题去"考"Agent，看它有没有把问题分发给正确的工具。
用法：
  1. 先启动服务：python -m uvicorn agent_api:app --port 8765
  2. 再运行本脚本：python eval_set.py
输出：每题的实际路由 vs 预期路由，最后给出总准确率。
为什么值钱：以后每次改代码（换提示词、加工具），跑一遍就知道有没有改坏——
这叫"回归测试"，是"我觉得还行"和"数据证明没改坏"的区别。
"""
import json
import urllib.request
import urllib.parse

BASE = "http://127.0.0.1:8765"

# ===== 评测集：20道题，expected 是"人类认为正确的路由" =====
EVAL_CASES = [
    # 计算（6题）
    ("帮我算一下 12*34", "CALCULATE"),
    ("计算 999除以3", "CALCULATE"),
    ("(45+55)*2 等于多少", "CALCULATE"),
    ("3的10次方是多少", "CALCULATE"),
    ("算一下 2026减去1999", "CALCULATE"),
    ("0.5乘以80的结果", "CALCULATE"),
    # RAG知识库（6题，知识库里只有：RAG / ReAct / 向量数据库 / Embedding）
    ("什么是RAG", "RAG"),
    ("什么是ReAct框架", "RAG"),
    ("向量数据库是干什么用的", "RAG"),
    ("Embedding是什么意思", "RAG"),
    ("介绍一下检索增强生成", "RAG"),
    ("知识库里怎么说智能体的", "RAG"),
    # 联网搜索（4题，明显需要实时信息）
    ("今天有什么AI新闻", "SEARCH"),
    ("查一下郑州今天的天气", "SEARCH"),
    ("现在最新的iPhone是什么型号", "SEARCH"),
    ("OpenAI最近发布了什么", "SEARCH"),
    # 直接回答（4题，不需要任何工具）
    ("你好", "DIRECT"),
    ("用一句话介绍你自己", "DIRECT"),
    ("写一句鼓励我学习的话", "DIRECT"),
    ("谢谢你", "DIRECT"),
]


def ask(msg: str):
    """调一次 /chat 接口，返回 (实际路由, 回答前60字)"""
    url = BASE + "/chat?msg=" + urllib.parse.quote(msg)
    with urllib.request.urlopen(url, timeout=90) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("route", "?"), (data.get("reply") or "")[:60]


def run():
    correct = 0
    wrong = []
    print(f"{'题号':<4}{'预期':<10}{'实际':<10}结果")
    print("-" * 50)
    for i, (msg, expected) in enumerate(EVAL_CASES, 1):
        actual, reply = ask(msg)
        ok = actual == expected
        if ok:
            correct += 1
        else:
            wrong.append((msg, expected, actual))
        mark = "✓" if ok else "✗"
        print(f"{i:<4}{expected:<10}{actual:<10}{mark}")

    total = len(EVAL_CASES)
    print("-" * 50)
    print(f"路由准确率: {correct}/{total} = {correct / total * 100:.1f}%")

    if wrong:
        print("\n【答错的题】")
        for msg, expected, actual in wrong:
            print(f"  「{msg}」预期 {expected}，实际走了 {actual}")

    # 按路由分组统计
    print("\n【补充：当前缓存与调用统计】")
    with urllib.request.urlopen(BASE + "/stats", timeout=10) as resp:
        print(json.dumps(json.loads(resp.read().decode("utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
