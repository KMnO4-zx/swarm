import inspect
from datetime import datetime


# debug_print 函数用于根据 debug 标志打印调试信息
def debug_print(debug: bool, *args: str) -> None:
    if not debug:
        # 如果 debug 为 False，直接返回，不打印信息
        return
    # 获取当前时间戳，格式为“年-月-日 时:分:秒”
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 将可变参数 args 转换为字符串并连接为一条消息
    message = " ".join(map(str, args))
    # 使用 ANSI 转义码设置控制台输出的颜色和格式，打印带有时间戳的调试信息
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


# merge_fields 函数用于合并字典中的字段
def merge_fields(target, source):
    # 遍历 source 字典中的所有键值对
    for key, value in source.items():
        # 如果 value 是字符串类型，直接将 source 中的值追加到 target 中
        if isinstance(value, str):
            target[key] += value
        # 如果 value 是字典且不为 None，递归调用 merge_fields 进行深度合并
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


# merge_chunk 函数用于合并来自增量更新的响应数据到最终的响应
def merge_chunk(final_response: dict, delta: dict) -> None:
    # 移除 delta 中的 "role" 键及其对应的值（如果存在）
    delta.pop("role", None)
    # 使用 merge_fields 函数合并 final_response 和 delta
    merge_fields(final_response, delta)

    # 获取 delta 中的工具调用（tool_calls）部分
    tool_calls = delta.get("tool_calls")
    # 如果 tool_calls 存在且包含元素
    if tool_calls and len(tool_calls) > 0:
        # 获取工具调用的索引
        index = tool_calls[0].pop("index")
        # 合并工具调用的相关信息到 final_response 的对应索引位置
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


# function_to_json 函数将一个 Python 函数转换为 JSON 序列化的字典形式
def function_to_json(func) -> dict:
    """
    将 Python 函数转换为 JSON 可序列化的字典，
    该字典描述了函数的签名，包括函数名称、描述和参数。

    参数:
        func: 要转换的函数对象。

    返回:
        描述函数签名的字典形式。
    """
    # 定义 Python 类型到 JSON 数据类型的映射
    type_map = {
        str: "string",       # 字符串类型映射为 JSON 的 "string"
        int: "integer",      # 整型类型映射为 JSON 的 "integer"
        float: "number",     # 浮点型映射为 JSON 的 "number"
        bool: "boolean",     # 布尔型映射为 JSON 的 "boolean"
        list: "array",       # 列表类型映射为 JSON 的 "array"
        dict: "object",      # 字典类型映射为 JSON 的 "object"
        type(None): "null",  # None 类型映射为 JSON 的 "null"
    }

    # 获取函数的签名信息
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        # 如果获取签名失败，则抛出异常并显示具体的错误信息
        raise ValueError(
            f"无法获取函数 {func.__name__} 的签名: {str(e)}"
        )

    # 用于存储参数信息的字典
    parameters = {}
    for param in signature.parameters.values():
        # 尝试获取参数的类型，如果无法找到对应的类型则默认设置为 "string"
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            # 如果参数类型不在 type_map 中，抛出异常并显示具体错误信息
            raise KeyError(
                f"未知的类型注解 {param.annotation}，参数名为 {param.name}: {str(e)}"
            )
        # 将参数名及其类型信息添加到参数字典中
        parameters[param.name] = {"type": param_type}

    # 获取函数中所有必需的参数（即没有默认值的参数）
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    # 返回包含函数描述信息的字典
    return {
        "type": "function",
        "function": {
            "name": func.__name__,            # 函数的名称
            "description": func.__doc__ or "", # 函数的文档字符串（如果不存在则为空字符串）
            "parameters": {
                "type": "object",
                "properties": parameters,     # 函数参数的类型描述
                "required": required,         # 必须参数的列表
            },
        },
    }

# 创建一个简单的函数 add_numbers
def add_numbers(a: int, b: int) -> int:
    """
    返回两个整数的和。

    参数：
        a: 第一个整数。
        b: 第二个整数。

    返回：
        两个整数的和。
    """
    return a + b

if __name__ == "__main__":

    # 使用 function_to_json 函数将 add_numbers 函数转换为 JSON 序列化的字典
    function_description = function_to_json(add_numbers)

    # 打印结果
    import json
    print(json.dumps(function_description, indent=4, ensure_ascii=False))