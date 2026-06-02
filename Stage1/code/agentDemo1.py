import operator
from datetime import datetime
from typing import Annotated, TypedDict, Union

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import settings


# ================= 1. 定义工具 (Tools) =================
@tool
def get_current_time():
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def multiply(a: int, b: int) -> int:
    """计算两个数字的乘积。"""
    return a * b


tools = [get_current_time, multiply]
tool_node = ToolNode(tools)


# ================= 2. 设置状态与模型 (State & LLM) =================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# 初始化模型并绑定工具
# 设置 timeout 和 max_retries 处理网络层面的错误
llm = ChatOpenAI(
    model=settings.openai_model_name,
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
    timeout=settings.timeout,
    max_retries=3
)
llm_with_tools = llm.bind_tools(tools)


# ================= 3. 定义节点逻辑 (Nodes & Logic) =================
def call_model(state: AgentState):
    """调用模型决策下一步动作"""
    response = llm_with_tools.invoke(state["messages"])
    # 可以在这里增加重试逻辑，如果模型输出格式不对
    return {"messages": [response]}


def should_continue(state: AgentState):
    """判断是继续执行工具还是结束"""
    last_message = state["messages"][-1]
    # 如果模型生成了 tool_calls，则进入 tools 节点
    if last_message.tool_calls:
        return "tools"
    # 否则直接结束，返回最终答案
    return END


# ================= 4. 构建工作流 (Workflow) =================
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# 设置入口
workflow.set_entry_point("agent")

# 设置条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# 从 tools 回到 agent，形成闭环 (Loop)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# ================= 5. 运行示例 (Execution) =================
if __name__ == "__main__":
    inputs = {"messages": [HumanMessage(content="现在几点了？然后计算 123 乘以 456 等于多少？")]}

    print("--- 开始 Agent 运行 ---")
    try:
        # recursion_limit: 设置最大步数（例如 10 步），防止无限循环
        for chunk in app.stream(inputs, config={"recursion_limit": 10}):
            for node, values in chunk.items():
                print(f"节点 [{node}] 运行结束")
                last_msg = values["messages"][-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    print(f"内容摘要: {last_msg.content}")
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"调用工具: {last_msg.tool_calls[0]['name']}")
            print("-" * 30)

    except Exception as e:
        print(f"运行中发生异常: {e}")