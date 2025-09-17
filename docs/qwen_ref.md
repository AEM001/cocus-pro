qwen-vl-plus-latest

下面是理解在线图像（通过URL指定，非本地图像）的示例代码。**后续有如何传入本地图像的示例代码。**
```import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-vl-max-latest", # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                    },
                },
                {"type": "text", "text": "图中描绘的是什么景象?"},
            ],
        },
    ],
)
print(completion.choices[0].message.content)
```



高分辨率图像理解
您可以通过设置vl_high_resolution_images参数为true，将通义千问VL模型的单图Token上限从1280提升至16384：（看具体情况而定）
| 参数值        | 单图 Token 上限 | 描述                                                         | 使用场景                                                     |
| ------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| True          | 16384           | - 表示模型的单图 Token 上限为 16384，超过该值的图像会被缩放，直到图像的 Token 小于 16384。<br>- 模型能直接处理更高像素的图像，能理解更多的图像细节。同时处理速度会降低，Token 用量也会增加 | 内容丰富、需要关注细节的场景                                 |
| False（默认值） | 1280            | - 表示模型的单图 Token 的上限为 1280，超过该值的图像会被缩放，直到图像的 Token 小于 1280。<br>- 模型的处理速度会提升，Token 用量较少 | 细节较少、只需用模型理解大致信息或对速度有较高要求的场景 |

vl_high_resolution_images参数仅支持DashScope SDK及HTTP方式下使用

```
import os
import dashscope

messages = [
    {
        "role": "user",
        "content": [
            {"image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250212/earbrt/vcg_VCG211286867973_RF.jpg"},
            {"text": "这张图表现了什么内容?"}
        ]
    }
]

response = dashscope.MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen-vl-max-latest',  # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    messages=messages,
    vl_high_resolution_images=True
)

print("大模型的回复:\n ",response.output.choices[0].message.content[0]["text"])
print("Token用量情况：","输入总Token：",response.usage["input_tokens"] , "，输入图像Token：" , response.usage["image_tokens"])
```

多图像输入
通义千问VL 模型支持单次请求传入多张图片进行综合分析，所有图像的总Token数需在模型的最大输入之内，可传入图像的最大数量请参考图像数量限制。

传入本地文件（Base64 编码或文件路径）
通义千问VL 提供两种本地文件上传方式：

Base64 编码上传

文件路径直接上传（传输更稳定，推荐方式）

直接向模型传入本地文件路径。仅 DashScope Python 和 Java SDK 支持，不支持 DashScope HTTP 和OpenAI 兼容方式。

请您参考下表，结合您的编程语言与操作系统指定文件的路径。

| 系统 | SDK | 传入的文件路径 | 示例 |
| ---- | ---- | ---- | ---- |
| Linux 或 macOS 系统 | Python SDK | file://{文件的绝对路径} | file:///home/images/test.png |


文件路径传入
```
import os
from dashscope import MultiModalConversation

# 将xxx/eagle.png替换为你本地图像的绝对路径
local_path = "xxx/eagle.png"
image_path = f"file://{local_path}"
messages = [{"role": "system",
                "content": [{"text": "You are a helpful assistant."}]},
                {'role':'user',
                'content': [{'image': image_path},
                            {'text': '图中描绘的是什么景象?'}]}]
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen-vl-max-latest',  # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    messages=messages)
print(response["output"]["choices"][0]["message"].content[0]["text"])
```

图像列表
```
import os

from dashscope import MultiModalConversation

local_path1 = "football1.jpg"
local_path2 = "football2.jpg"
local_path3 = "football3.jpg"
local_path4 = "football4.jpg"

image_path1 = f"file://{local_path1}"
image_path2 = f"file://{local_path2}"
image_path3 = f"file://{local_path3}"
image_path4 = f"file://{local_path4}"

messages = [{"role": "system",
                "content": [{"text": "You are a helpful assistant."}]},
                {'role':'user',
                # 若模型属于Qwen2.5-VL系列且传入图像列表时，可设置fps参数，表示图像列表是由原视频每隔 1/fps 秒抽取的，其他模型设置则不生效
                'content': [{'video': [image_path1,image_path2,image_path3,image_path4],"fps":2},
                            {'text': '这段视频描绘的是什么景象?'}]}]
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen-vl-max-latest',  # 此处以qwen-vl-max-latest为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
    messages=messages)

print(response["output"]["choices"][0]["message"].content[0]["text"])
```

使用SDK调用时需配置的base_url：https://dashscope.aliyuncs.com/compatible-mode/v1

使用HTTP方式调用时需配置的endpoint：POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

您需要已获取API Key并配置API Key到环境变量。如果通过OpenAI SDK进行调用，还需要安装SDK。
import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-vl-plus",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[{"role": "user","content": [
            {"type": "image_url",
             "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
            {"type": "text", "text": "这是什么"},
            ]}]
    )
print(completion.model_dump_json())


model string （必选）



model string （必选）

模型名称。

支持的模型：通义千问大语言模型（商业版、开源版）、通义千问VL、代码模型、通义千问Omni、数学模型。

通义千问Audio暂不支持OpenAI兼容模式，仅支持DashScope方式。
具体模型名称和计费，请参见模型列表。

messages array （必选）

由历史对话组成的消息列表。

消息类型

System Message object （可选）

模型的目标或角色。如果设置系统消息，请放在messages列表的第一位。

属性

QwQ 模型不建议设置 System Message，QVQ 模型设置System Message不会生效。
User Message object （必选）

用户发送给模型的消息。

属性

Assistant Message object （可选）

模型对用户消息的回复。

属性

Tool Message object （可选）

工具的输出信息。


