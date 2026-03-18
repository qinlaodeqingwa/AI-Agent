# ============================================================
# agent.py — 第一个 LangGraph ReAct Agent
# 功能：读取 Prometheus 指标，分析服务健康状态
# ============================================================

import httpx
import json
import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
load_dotenv()


# ══════════════════════════════════════════════════════════════
# 第一部分：定义 State（Agent 的共享记忆）
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    Agent 的状态定义。

    messages: 对话历史列表
    Annotated[..., add_messages] 的含义：
      每次更新 messages 时，不是"替换"而是"追加"
      这样对话历史就会一直保留
    """
    messages: Annotated[list, add_messages]


# ══════════════════════════════════════════════════════════════
# 第二部分：定义工具（Tools）
# Agent 通过这些工具与外部世界交互
# ══════════════════════════════════════════════════════════════

PROMETHEUS_URL = "http://47.100.218.182:9091/metrics"  # prometheus服务地址


@tool
def query_prometheus_metrics() -> str:
    """
    查询 Prometheus 指标原始数据。
    返回当前服务的所有监控指标。
    当需要了解服务健康状态时调用此工具。
    """
    try:
        response = httpx.get(PROMETHEUS_URL, timeout=5.0)
        if response.status_code == 200:
            return response.text
        else:
            return f"查询失败，状态码：{response.status_code}"
    except Exception as e:
        return f"连接失败：{str(e)}，请确保第二课的服务正在运行"


@tool
def analyze_error_rate(metrics_text: str) -> str:
    """
    分析错误率。
    从指标文本中提取错误相关数据并计算错误率。

    参数：
        metrics_text: Prometheus 指标原始文本
    """
    lines = metrics_text.split('\n')

    total_requests = 0
    error_requests = 0

    for line in lines:
        # 跳过注释行
        if line.startswith('#') or not line.strip():
            continue

        # 统计总请求数
        if 'http_requests_total' in line and 'status_code' in line:
            try:
                value = float(line.split(' ')[-1])
                total_requests += value
                # 4xx 和 5xx 算错误
                if 'status_code="4' in line or 'status_code="5' in line:
                    error_requests += value
            except:
                pass

    if total_requests == 0:
        return "暂无请求数据，请先向第二课的服务发送一些请求"

    error_rate = (error_requests / total_requests) * 100

    result = {
        "总请求数": total_requests,
        "错误请求数": error_requests,
        "错误率": f"{error_rate:.2f}%",
        "健康状态": "⚠️ 需要关注" if error_rate > 10 else "✅ 正常"
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_slow_endpoints(metrics_text: str) -> str:
    """
    找出响应慢的接口。
    分析 Histogram 数据，找出 P95 延迟高的接口。

    参数：
        metrics_text: Prometheus 指标原始文本
    """
    lines = metrics_text.split('\n')

    # 收集每个接口的请求总时间和总次数
    endpoint_sum = {}  # 总耗时
    endpoint_count = {}  # 总次数

    for line in lines:
        if line.startswith('#') or not line.strip():
            continue

        # 解析 _sum（总耗时）
        if 'http_request_duration_seconds_sum' in line:
            try:
                # 提取 endpoint 标签值
                endpoint = line.split('endpoint="')[1].split('"')[0]
                value = float(line.split(' ')[-1])
                endpoint_sum[endpoint] = value
            except:
                pass

        # 解析 _count（总次数）
        if 'http_request_duration_seconds_count' in line:
            try:
                endpoint = line.split('endpoint="')[1].split('"')[0]
                value = float(line.split(' ')[-1])
                endpoint_count[endpoint] = value
            except:
                pass

    if not endpoint_sum:
        return "暂无延迟数据"

    results = []
    for endpoint in endpoint_sum:
        if endpoint in endpoint_count and endpoint_count[endpoint] > 0:
            avg_latency = endpoint_sum[endpoint] / endpoint_count[endpoint]
            results.append({
                "接口": endpoint,
                "平均响应时间": f"{avg_latency * 1000:.1f}ms",
                "请求次数": int(endpoint_count[endpoint]),
                "状态": "🐢 慢接口" if avg_latency > 0.3 else "⚡ 正常"
            })

    # 按平均延迟排序
    results.sort(
        key=lambda x: float(x["平均响应时间"].replace("ms", "")),
        reverse=True
    )

    return json.dumps(results, ensure_ascii=False, indent=2)


# 把所有工具放进列表
tools = [query_prometheus_metrics, analyze_error_rate, get_slow_endpoints]

# ══════════════════════════════════════════════════════════════
# 第三部分：定义 LLM 和 Agent 节点
# ══════════════════════════════════════════════════════════════

# 初始化 LLM，并绑定工具
# bind_tools 告诉 LLM："你有这些工具可以用"
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    model="deepseek-chat",
)
llm_with_tools = llm.bind_tools(tools)

# 系统提示词：定义 Agent 的角色和行为
SYSTEM_PROMPT = """你是一个专业的 SRE（站点可靠性工程师）AI 助手。
你的职责是分析 Prometheus 监控指标，发现服务问题并给出建议。

工作流程：
1. 首先调用 query_prometheus_metrics 获取原始指标
2. 根据用户问题，选择合适的分析工具
3. 综合分析结果，给出清晰的中文报告

报告格式要求：
- 用 emoji 标注健康状态
- 给出具体数字
- 提供可操作的建议
"""


def agent_node(state: AgentState) -> AgentState:
    """
    Agent 节点：调用 LLM 进行推理和决策。

    这是 ReAct 中的 "Reasoning" 部分：
    LLM 看到当前对话历史，决定：
      - 直接回答用户
      - 还是调用某个工具获取更多信息
    """
    # 在消息列表最前面加入系统提示
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # 调用 LLM（LLM 可能返回普通文本，也可能返回"我要调用工具X"）
    response = llm_with_tools.invoke(messages)

    # 把 LLM 的回复追加到 messages 里
    return {"messages": [response]}


# ══════════════════════════════════════════════════════════════
# 第四部分：定义路由逻辑（条件边）
# ══════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> str:
    """
    条件边：决定下一步去哪个节点。

    检查 LLM 的最新回复：
    - 如果包含 tool_calls（LLM 想调工具）→ 去 tools 节点执行工具
    - 如果不包含 tool_calls（LLM 直接给出答案）→ 结束

    这就是 ReAct 循环的"出口判断"
    """
    last_message = state["messages"][-1]

    # 检查是否有工具调用请求
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"  # → 去执行工具
    else:
        return END  # → 结束，返回最终答案


# ══════════════════════════════════════════════════════════════
# 第五部分：构建图（Graph）
# 把节点和边组装成完整的 Agent
# ══════════════════════════════════════════════════════════════

def build_agent():
    """构建并编译 LangGraph Agent"""

    # 1. 创建图，指定 State 类型
    graph = StateGraph(AgentState)

    # 2. 添加节点
    graph.add_node("agent", agent_node)  # LLM 推理节点
    graph.add_node("tools", ToolNode(tools))  # 工具执行节点
    #   ToolNode 是 LangGraph 内置的工具执行器
    #   它会自动解析 LLM 的 tool_calls，找到对应工具并执行

    # 3. 设置入口节点（从哪里开始）
    graph.set_entry_point("agent")

    # 4. 添加条件边（agent 节点之后的分支）
    graph.add_conditional_edges(
        "agent",  # 从 agent 节点出发
        should_continue,  # 用这个函数决定去哪
        {
            "tools": "tools",  # 返回 "tools" → 去 tools 节点
            END: END  # 返回 END → 结束
        }
    )

    # 5. 添加普通边（tools 执行完后，永远回到 agent 节点继续推理）
    graph.add_edge("tools", "agent")
    #   这就形成了 ReAct 的循环：
    #   agent → tools → agent → tools → ... → END

    # 6. 编译图（生成可执行的 Agent）
    return graph.compile()


# ══════════════════════════════════════════════════════════════
# 第六部分：运行 Agent
# ══════════════════════════════════════════════════════════════

def run_agent(question: str):
    """运行 Agent 并打印结果"""
    agent = build_agent()

    print(f"\n{'=' * 50}")
    print(f"🤔 用户问题：{question}")
    print(f"{'=' * 50}\n")

    # 流式输出，实时看到 Agent 的每一步
    for step in agent.stream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="values"
    ):
        last_message = step["messages"][-1]

        # 打印每一步的动作
        msg_type = type(last_message).__name__

        if msg_type == "AIMessage":
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    print(f"🔧 调用工具：{tc['name']}")
                    if tc.get('args'):
                        # 只打印参数的前100个字符，避免太长
                        args_str = str(tc['args'])[:100]
                        print(f"   参数：{args_str}...")
            else:
                print(f"\n📊 最终分析报告：")
                print(last_message.content)

        elif msg_type == "ToolMessage":
            print(f"✅ 工具返回：{last_message.content[:200]}...")


# ══════════════════════════════════════════════════════════════
# 主程序入口
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os

    # 设置 OpenAI API Key
    # 建议用环境变量，不要硬编码在代码里
    # os.environ["OPENAI_API_KEY"] = "你的key"

    # 测试问题1：整体健康检查
    run_agent("帮我分析一下当前服务的健康状态，包括错误率和响应速度")

    # 测试问题2：专项分析
    # run_agent("哪个接口最慢？需要优化吗？")
