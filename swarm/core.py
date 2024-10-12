# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List, Callable, Union

# Package/library imports
from openai import OpenAI


# Local imports
from .util import function_to_json, debug_print, merge_chunk
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    def __init__(self, client=None):
        if not client:
            client = OpenAI()
        self.client = client

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        # 获取聊天完成的响应
        # 参数包括代理对象、消息历史、上下文变量、模型选择、是否流式输出和调试模式
        context_variables = defaultdict(str, context_variables)  # 将上下文变量转换为默认字典
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )  # 获取代理的指令信息，可能是可调用的函数
        messages = [{"role": "system", "content": instructions}] + history  # 构建消息列表，系统消息和历史消息
        debug_print(debug, "Getting chat completion for...:", messages)  # 如果调试模式开启，打印调试信息

        tools = [function_to_json(f) for f in agent.functions]  # 将代理的功能函数转换为 JSON 格式
        # 隐藏上下文变量，不传递给模型
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)  # 移除上下文变量
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)  # 从必需参数中移除上下文变量

        create_params = {
            "model": model_override or agent.model,  # 使用指定的模型或代理模型
            "messages": messages,  # 传递的消息历史
            "tools": tools or None,  # 工具函数
            "tool_choice": agent.tool_choice,  # 工具选择
            "stream": stream,  # 是否流式输出
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls  # 如果有工具，允许并行调用工具

        return self.client.chat.completions.create(**create_params)  # 创建聊天完成的响应

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result: # 如果结果是 Result 对象，直接返回
                return result

            case Agent() as agent: # 如果结果是 agent，返回包含代理名称的结果对象
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))  # 尝试将其他类型结果转换为字符串
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        # 处理工具函数调用
        function_map = {f.__name__: f for f in functions}  # 创建功能函数的映射，函数名作为键
        partial_response = Response(messages=[], agent=None, context_variables={})  # 初始化部分响应对象

        for tool_call in tool_calls:
            name = tool_call.function.name  # 获取工具函数的名称
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")  # 如果工具函数不存在，打印调试信息
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",  # 添加错误消息
                    }
                )
                continue  # 跳过这个工具函数

            args = json.loads(tool_call.function.arguments)  # 解析工具函数的参数
            debug_print(debug, f"Processing tool call: {name} with arguments {args}")  # 打印调试信息

            func = function_map[name]  # 获取对应的功能函数
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables  # 如果函数需要上下文变量，将其加入参数
            raw_result = function_map[name](**args)  # 调用功能函数并获取原始结果

            result: Result = self.handle_function_result(raw_result, debug)  # 处理功能函数的返回结果
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,  # 将工具函数的返回值加入响应消息中
                }
            )
            partial_response.context_variables.update(result.context_variables)  # 更新上下文变量
            if result.agent:
                partial_response.agent = result.agent  # 如果返回了新的代理对象，更新代理

        return partial_response  # 返回部分响应

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        # 运行并流式输出聊天完成结果
        active_agent = agent  # 当前激活的代理对象
        context_variables = copy.deepcopy(context_variables)  # 深拷贝上下文变量
        history = copy.deepcopy(messages)  # 深拷贝消息历史
        init_len = len(messages)  # 初始消息的长度

        while len(history) - init_len < max_turns:  # 在消息数未超过最大轮次的情况下，循环执行

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )  # 获取聊天完成的流式响应

            yield {"delim": "start"}  # 开始流式响应
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())  # 解析流式响应中的每个数据块
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name  # 设置发送者为当前代理对象
                yield delta  # 返回解析后的数据块
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)  # 合并数据块到消息中
            yield {"delim": "end"}  # 结束流式响应

            message["tool_calls"] = list(message.get("tool_calls", {}).values())  # 获取工具调用信息
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)  # 打印调试信息，显示接收到的消息
            history.append(message)  # 将消息添加到历史记录中

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")  # 如果没有工具调用或不执行工具，结束回合
                break

            # 处理工具调用，更新上下文变量，并切换代理对象
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)  # 将部分响应的消息添加到历史中
            context_variables.update(partial_response.context_variables)  # 更新上下文变量
            if partial_response.agent:
                active_agent = partial_response.agent  # 如果有新的代理对象，更新当前代理

        yield {
            "response": Response(
                messages=history[init_len:],  # 返回最终的消息历史
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        # 非流式运行聊天完成的主方法
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )  # 如果启用流式，调用流式方法

        active_agent = agent
        context_variables = copy.deepcopy(context_variables)  # 深拷贝上下文变量
        history = copy.deepcopy(messages)  # 深拷贝消息历史
        init_len = len(messages)  # 记录初始消息的长度

        while len(history) - init_len < max_turns and active_agent:  # 如果未达到最大轮次且有激活代理，继续循环
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )  # 获取聊天完成

            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)  # 打印调试信息
            message.sender = active_agent.name  # 设置消息的发送者为当前代理
            history.append(json.loads(message.model_dump_json()))  # 将消息转换为 JSON 格式并添加到历史记录中

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")  # 如果没有工具调用或不执行工具，结束回合
                break

            # 处理工具调用，更新上下文变量，并切换代理对象
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)  # 将部分响应的消息添加到历史中
            context_variables.update(partial_response.context_variables)  # 更新上下文变量
            if partial_response.agent:
                active_agent = partial_response.agent  # 如果有新的代理对象，更新当前代理

        return Response(
            messages=history[init_len:],  # 返回最终的消息历史
            agent=active_agent,
            context_variables=context_variables,
        )
