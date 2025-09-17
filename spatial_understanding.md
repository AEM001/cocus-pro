# 物体定位

仅 Qwen2.5-VL 模型支持以 Box 定位或 Point 定位的两种方式对物体定位，以 Box 定位方式会返回矩形框的左上角和右下角的坐标，以 Point 定位的方式会返回矩形框中心点的坐标（两类坐标均相对于缩放后的图像左上角的绝对值，单位为像素）。

模型在进行图像理解前会对图像进行缩放处理，您可以参考 Qwen2.5-VL 中的代码将坐标映射到原图中，同时您还可以通过设置 vl_high_resolution_images 参数为 True 来尽可能保证返回图像不被缩放，但可能会带来 Token 的消耗。  
Qwen2.5-VL 模型 480*480～2560*2560 分辨率范围内，物体定位效果较为鲁棒，在此范围之外可能会偶发 bbox 漂移现象。

---

## 1. Prompt 技巧

| 定位方式   | 支持的输出方式 | 推荐 Prompt |
| ---------- | -------------- | ----------- |
| Box 定位   | JSON 或纯文本  | 检测图中所有{物体}并以{JSON/纯文本}格式输出其 bbox 的坐标 |
| Point 定位 | JSON 或 XML    | 以点的形式定位图中所有{物体}，以{JSON/XML}格式输出其 point 坐标 |

---

## 2. Prompt 改进思路

- 当检测密集排列的物体时，如 Prompt 为“检测图中所有人”，模型可能会混淆了“每个人”和“所有人”的语义，从而仅输出将所有人物包含在内的框，可以通过下列提示词间接增强检测每个对象：

  - **Box 定位**：定位图中每一个{某类物体}并描述其各自的{某种特征}，以{JSON/纯文本}格式输出其 bbox 坐标。  
  - **Point 定位**：以点的形式定位图中每一个{某类物体}并描述各自的{某种特征}，以{JSON/XML}格式输出其 point 坐标。

- 定位结果中可能会出现 ```json``` 或者 ```xml``` 等无关内容，可在 Prompt 中明确禁止该内容输出，如“请你以 JSON 格式输出，不要输出 ```json``` 代码段”。
## 示例
示例：
Box定位：
提示词：定位每一个蛋糕的位置，并描述其各自的特征，以JSON格式输出所有的bbox的坐标，不要输出```json```代码段。
示例代码：
curl -X POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
    "model": "qwen-vl-max-latest",
    "input":{
        "messages":[
            {
                "role": "user",
                "content": [
                    {"image": "https://img.alicdn.com/imgextra/i3/O1CN01I1CXf21UR0Ld20Yzs_!!6000000002513-2-tps-1024-1024.png"},
                    {"text":  "用一个个框定位图像每一个蛋糕的位置并描述其各自的特征，以JSON格式输出所有的bbox的坐标，不要输出```json```代码段"}
                ]
            }
        ],
        "vl_high_resolution_images":"True",
        "temperature":"0",
        "top_k":"1",
        "seed":"3407"
    }
}'

输出示例：
[
  {
    "bbox": [60, 395, 204, 578],
    "description": "巧克力蛋糕，顶部覆盖红色糖霜和彩色糖粒"
  },
  {
    "bbox": [248, 381, 372, 542],
    "description": "粉色糖霜的蛋糕，顶部有白色和蓝色的糖粒"
  },
  {
    "bbox": [400, 368, 504, 504],
    "description": "粉色糖霜的蛋糕，顶部有白色和蓝色的糖粒"
  },
  {
    "bbox": [530, 355, 654, 526],
    "description": "粉色糖霜的蛋糕，顶部有白色和蓝色的糖粒"
  },
  {
    "bbox": [432, 445, 566, 606],
    "description": "粉红色糖霜的蛋糕，顶部有两个黑色眼睛"
  },
  {
    "bbox": [630, 475, 774, 646],
    "description": "黄色糖霜的蛋糕，顶部有多种颜色的糖粒"
  },
  {
    "bbox": [740, 380, 868, 539],
    "description": "巧克力蛋糕，顶部覆盖棕色糖霜"
  },
  {
    "bbox": [796, 512, 960, 693],
    "description": "黄色糖霜的蛋糕，顶部有多种颜色的糖粒"
  },
  {
    "bbox": [39, 555, 200, 736],
    "description": "黄色糖霜的蛋糕，顶部有多种颜色的糖粒"
  },
  {
    "bbox": [292, 546, 446, 707],
    "description": "黑色蛋糕，顶部有白色糖霜和两个黑色眼睛"
  },
  {
    "bbox": [516, 564, 666, 715],
    "description": "黄色糖霜的蛋糕，顶部有两个黑色眼睛"
  },
  {
    "bbox": [352, 655, 516, 822],
    "description": "白色糖霜的蛋糕，顶部有两个黑色眼睛"
  },
  {
    "bbox": [130, 746, 304, 924],
    "description": "白色糖霜的蛋糕，顶部有两个黑色眼睛"
  }
]

Point定位：
提示词：以点的形式定位图中见义勇为的人，并以XML格式输出结果，不要输出```xml```代码段。
代码示例：
curl -X POST https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation \
-H "Authorization: Bearer $DASHSCOPE_API_KEY" \
-H 'Content-Type: application/json' \
-d '{
    "model": "qwen-vl-max-latest",
    "input":{
        "messages":[
            {
                "role": "user",
                "content": [
                    {"image": "https://img.alicdn.com/imgextra/i1/O1CN01ILRlNK1gvU5xqbaxb_!!6000000004204-49-tps-1138-640.webp"},
                    {"text":  "以点的形式定位图中见义勇为的人，并以XML格式输出结果，不要输出```xml```代码段。"}
                ]
            }
        ],
        "vl_high_resolution_images":"True",
        "temperature":"0",
        "top_k":"1",
        "seed":"3407"
    }
}'

输出示例：

< points x1 = "284"
y1 = "305"
alt = "见义勇为的人" > 见义勇为的人 < /points>


# Spatial Understanding with Qwen2.5-VL

This notebook showcases Qwen2.5-VL's advanced spatial localization abilities, including accurate object detection and specific target grounding within images. 

See how it integrates visual and linguistic understanding to interpret complex scenes effectively.

Prepare the environment


```python
!pip install git+https://github.com/huggingface/transformers
!pip install qwen-vl-utils
!pip install openai
```

#### \[Setup\]

Load visualization utils.


```python
# @title Plotting Util

# Get Noto JP font to display janapese characters
!apt-get install fonts-noto-cjk  # For Noto Sans CJK JP

#!apt-get install fonts-source-han-sans-jp # For Source Han Sans (Japanese)

import json
import random
import io
import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import xml.etree.ElementTree as ET

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def decode_xml_points(text):
    try:
        root = ET.fromstring(text)
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            points.append([x, y])
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        print(e)
        return None

def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
      json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
      end_idx = bounding_boxes.rfind('"}') + len('"}')
      truncated_text = bounding_boxes[:end_idx] + "]"
      json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
      # Select a color from the list
      color = colors[i % len(colors)]

      # Convert normalized coordinates to absolute coordinates
      abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
      abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
      abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
      abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

      if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

      if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y1

      # Draw the bounding box
      draw.rectangle(
          ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
      )

      # Draw the text
      if "label" in bounding_box:
        draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    img.show()


def plot_points(im, text, input_width, input_height):
  img = im
  width, height = img.size
  draw = ImageDraw.Draw(img)
  colors = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver',
  ] + additional_colors
  xml_text = text.replace('```xml', '')
  xml_text = xml_text.replace('```', '')
  data = decode_xml_points(xml_text)
  if data is None:
    img.show()
    return
  points = data['points']
  description = data['phrase']

  font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

  for i, point in enumerate(points):
    color = colors[i % len(colors)]
    abs_x1 = int(point[0])/input_width * width
    abs_y1 = int(point[1])/input_height * height
    radius = 2
    draw.ellipse([(abs_x1 - radius, abs_y1 - radius), (abs_x1 + radius, abs_y1 + radius)], fill=color)
    draw.text((abs_x1 + 8, abs_y1 + 6), description, fill=color, font=font)
  
  img.show()
  

# @title Parsing JSON output
def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


    正在读取软件包列表... 完成%
    正在分析软件包的依赖关系树... 完成%
    正在读取状态信息... 完成                   
    fonts-noto-cjk 已经是最新版 (1:20220127+repack1-1)。
    升级了 0 个软件包，新安装了 0 个软件包， 要卸载 0 个软件包，有 177 个软件包未被升级。


Load model and processors.


```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",device_map="auto")
processor = AutoProcessor.from_pretrained(model_path)
```

    /opt/conda/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    2025-01-29 19:51:29.636861: I tensorflow/core/util/port.cc:111] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2025-01-29 19:51:29.639620: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2025-01-29 19:51:29.671029: E tensorflow/compiler/xla/stream_executor/cuda/cuda_dnn.cc:9342] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    2025-01-29 19:51:29.671058: E tensorflow/compiler/xla/stream_executor/cuda/cuda_fft.cc:609] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    2025-01-29 19:51:29.671088: E tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.cc:1518] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
    2025-01-29 19:51:29.677862: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2025-01-29 19:51:29.678441: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2025-01-29 19:51:30.519804: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    Loading checkpoint shards: 100%|██████████| 5/5 [00:10<00:00,  2.01s/it]


Load inference function.


```python
def inference(img_url, prompt, system_prompt="You are a helpful assistant", max_new_tokens=1024):
  image = Image.open(img_url)
  messages = [
    {
      "role": "system",
      "content": system_prompt
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": prompt
        },
        {
          "image": img_url
        }
      ]
    }
  ]
  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  print("input:\n",text)
  inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')

  output_ids = model.generate(**inputs, max_new_tokens=1024)
  generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
  output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  print("output:\n",output_text[0])

  input_height = inputs['image_grid_thw'][0][1]*14
  input_width = inputs['image_grid_thw'][0][2]*14

  return output_text[0], input_height, input_width
```

inference function with API


```python
from openai import OpenAI
import os
import base64
#  base 64 编码格式
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# @title inference function with API
def inference_with_api(image_path, prompt, sys_prompt="You are a helpful assistant.", model_id="qwen2.5-vl-72b-instruct", min_pixels=512*28*28, max_pixels=2048*28*28):
    base64_image = encode_image(image_path)
    client = OpenAI(
        #If the environment variable is not configured, please replace the following line with the Dashscope API Key: api_key="sk-xxx".
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )


    messages=[
        {
            "role": "system",
            "content": [{"type":"text","text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # Pass in BASE64 image data. Note that the image format (i.e., image/{format}) must match the Content Type in the list of supported images. "f" is the method for string formatting.
                    # PNG image:  f"data:image/png;base64,{base64_image}"
                    # JPEG image: f"data:image/jpeg;base64,{base64_image}"
                    # WEBP image: f"data:image/webp;base64,{base64_image}"
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    completion = client.chat.completions.create(
        model = model_id,
        messages = messages,
       
    )
    return completion.choices[0].message.content
```

#### 1. Detect certain object in the image

Let's start with a simple scenario where we want to locate certain objects in an image.

Besides, we can further prompt the model to describe their unique characteristics or features by explicitly giving that order.


```python
image_path = "./assets/spatial_understanding/cakes.png"


## Use a local HuggingFace model to inference.
# prompt in chinese
prompt = "框出每一个小蛋糕的位置，以json格式输出所有的坐标"
# prompt in english
prompt = "Outline the position of each small cake and output all the coordinates in JSON format."
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
print(image.size)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image,response,input_width,input_height)




## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)

```

    input:
     <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Outline the position of each small cake and output all the coordinates in JSON format.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
    <|im_start|>assistant
    
    output:
     ```json
    [
        {"bbox_2d": [139, 768, 315, 954], "label": "small cake"},
        {"bbox_2d": [366, 679, 536, 849], "label": "small cake"},
        {"bbox_2d": [44, 575, 204, 753], "label": "small cake"},
        {"bbox_2d": [310, 567, 459, 729], "label": "small cake"},
        {"bbox_2d": [533, 579, 689, 738], "label": "small cake"},
        {"bbox_2d": [652, 492, 799, 662], "label": "small cake"},
        {"bbox_2d": [829, 526, 995, 716], "label": "small cake"},
        {"bbox_2d": [765, 393, 895, 552], "label": "small cake"},
        {"bbox_2d": [551, 369, 676, 542], "label": "small cake"},
        {"bbox_2d": [411, 381, 519, 520], "label": "small cake"},
        {"bbox_2d": [262, 392, 384, 558], "label": "small cake"},
        {"bbox_2d": [69, 408, 212, 593], "label": "small cake"}
    ]
    ```
    (1024, 1024)
    (640, 640)



    
![png](spatial_understanding_files/spatial_understanding_12_1.png)
    


#### 2. Detect a specific object using descriptions

Further, you can search for a specific object by using a short phrase or sentence to describe it.


```python
image_path = "./assets/spatial_understanding/cakes.png"

# prompt in chinses
prompt = "定位最右上角的棕色蛋糕，以JSON格式输出其bbox坐标"
# prompt in english
prompt = "Locate the top right brown cake, output its bbox coordinates using JSON format."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image,response,input_width,input_height)

## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)
```

    input:
     <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Locate the top right brown cake, output its bbox coordinates using JSON format.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
    <|im_start|>assistant
    
    output:
     ```json
    [
    	{"bbox_2d": [765, 394, 891, 550], "label": "top right brown cake"}
    ]
    ```
    (640, 640)



    
![png](spatial_understanding_files/spatial_understanding_14_1.png)
    


#### 3. Point to certain objects in xml format

In addition to the above mentioned bbox format [x1, y1, x2, y2], Qwen2.5-VL also supports point-based grounding. You can point to a specific object and the model is trained to output xml-style results.


```python
image_path = "./assets/spatial_understanding/cakes.png"

# prompt in chinese
prompt = "以点的形式定位图中桌子远处的擀面杖，以XML格式输出其坐标"
# prompt in english
prompt = "point to the rolling pin on the far side of the table, output its coordinates in XML format <points x y>object</points>"

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_points(image, response, input_width, input_height)

## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_points(image, response, input_width, input_height)
```


    
![png](spatial_understanding_files/spatial_understanding_16_0.png)
    


#### 4. Reasoning capability


```python
image_path = "./assets/spatial_understanding/Origamis.jpg"

# prompt in chinese
prompt = "框出图中纸狐狸的影子，以json格式输出其bbox坐标"
# prompt in english
prompt = "Locate the shadow of the paper fox, report the bbox coordinates in JSON format."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image, response, input_width, input_height)

## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)
```

    input:
     <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Locate the shadow of the paper fox, report the bbox coordinates in JSON format.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
    <|im_start|>assistant
    
    output:
     ```json
    [
    	{"bbox_2d": [1098, 1304, 1576, 1900], "label": "shadow of the paper fox"}
    ]
    ```
    (640, 482)



    
![png](spatial_understanding_files/spatial_understanding_18_1.png)
    


#### 5. Understand relationships across different instances


```python
image_path = "./assets/spatial_understanding/cartoon_brave_person.jpeg"

# prompt in chinese
prompt = "框出图中见义勇为的人，以json格式输出其bbox坐标"
# prompt in english
prompt = "Locate the person who act bravely, report the bbox coordinates in JSON format."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image, response, input_width, input_height)


## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)
```

    input:
     <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Locate the person who act bravely, report the bbox coordinates in JSON format.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
    <|im_start|>assistant
    
    output:
     ```json
    [
    	{"bbox_2d": [74, 58, 526, 623], "label": "person who acts bravely"}
    ]
    ```
    (640, 360)



    
![png](spatial_understanding_files/spatial_understanding_20_1.png)
    


#### 6. Find a special instance with unique characteristic (color, location, utility, ...)


```python
url = "./assets/spatial_understanding/multiple_items.png"

# prompt in chinese
prompt = "如果太阳很刺眼，我应该用这张图中的什么物品，框出该物品在图中的bbox坐标，并以json格式输出"
# prompt in english
prompt = "If the sun is very glaring, which item in this image should I use? Please locate it in the image with its bbox coordinates and its name and output in JSON format."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(url, prompt)

image = Image.open(url)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image, response, input_width, input_height)


## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)
```

    input:
     <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    If the sun is very glaring, which item in this image should I use? Please locate it in the image with its bbox coordinates and its name and output in JSON format.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
    <|im_start|>assistant
    
    output:
     ```json
    [
    	{"bbox_2d": [150, 196, 308, 310], "label": "sunglasses"}
    ]
    ```
    (640, 465)



    
![png](spatial_understanding_files/spatial_understanding_22_1.png)
    


#### 7. Use Qwen2.5-VL grounding capabilities to help counting


```python
image_path = "./assets/spatial_understanding/multiple_items.png"

# prompt in chinese
prompt = "请以JSON格式输出图中所有物体bbox的坐标以及它们的名字，然后基于检测结果回答以下问题：图中物体的数目是多少？"
# prompt in english
prompt = "Please first output bbox coordinates and names of every item in this image in JSON format, and then answer how many items are there in the image."

## Use a local HuggingFace model to inference.
response, input_height, input_width = inference(image_path, prompt)

image = Image.open(image_path)
image.thumbnail([640,640], Image.Resampling.LANCZOS)
plot_bounding_boxes(image,response,input_width,input_height)

# # Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# min_pixels = 512*28*28
# max_pixels = 2048*28*28
# image = Image.open(image_path)
# width, height = image.size
# input_height,input_width = smart_resize(height,width,min_pixels=min_pixels, max_pixels=max_pixels)
# response = inference_with_api(image_path, prompt, min_pixels=min_pixels, max_pixels=max_pixels)
# plot_bounding_boxes(image, response, input_width, input_height)
```

    input:
     <|im_start|>system
    You are a helpful assistant<|im_end|>
    <|im_start|>user
    Please first output bbox coordinates and names of every item in this image in JSON format, and then answer how many items are there in the image.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
    <|im_start|>assistant
    
    output:
     ```json
    [
        {"bbox_2d": [25, 6, 87, 124], "label": "ice cream"},
        {"bbox_2d": [169, 3, 288, 130], "label": "flip-flops"},
        {"bbox_2d": [348, 0, 456, 124], "label": "beach umbrella"},
        {"bbox_2d": [519, 5, 634, 124], "label": "starfish"},
        {"bbox_2d": [14, 182, 118, 297], "label": "cocktail drink"},
        {"bbox_2d": [149, 196, 308, 283], "label": "sunglasses"},
        {"bbox_2d": [353, 196, 467, 300], "label": "lifebuoy"},
        {"bbox_2d": [530, 196, 644, 297], "label": "watermelon slice"},
        {"bbox_2d": [2, 383, 134, 476], "label": "hat"},
        {"bbox_2d": [168, 336, 308, 460], "label": "palm tree"},
        {"bbox_2d": [353, 364, 467, 476], "label": "sun"},
        {"bbox_2d": [548, 342, 644, 476], "label": "martini glass"}
    ]
    ```
    
    There are 12 items in the image.
    (640, 465)



    
![png](spatial_understanding_files/spatial_understanding_24_1.png)
    


#### 8. spatial understanding with designed system prompt
The above usage is based on the default system prompt. You can also change the system prompt to obtain other output format like plain text.
Qwen2.5-VL now support these formats:
* bbox-format: JSON

`{"bbox_2d": [x1, y1, x2, y2], "label": "object name/description"}`

* bbox-format: plain text

`x1,y1,x2,y2 object_name/description`

* point-format: XML

`<points x y>object_name/description</points>`

* point-format: JSON

`{"point_2d": [x, y], "label": "object name/description"}`

Change your system prompt to use plain text as output format


```python
image_path = "./assets/spatial_understanding/cakes.png"
image = Image.open(image_path)
system_prompt = "As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'."
prompt = "find all cakes"
response, input_height, input_width = inference(image_path, prompt, system_prompt=system_prompt)



## Use an API-based approach to inference. Apply API key here: https://bailian.console.alibabacloud.com/?apiKey=1
# from qwen_vl_utils import smart_resize
# os.environ['DASHSCOPE_API_KEY'] = 'your_api_key_here' 
# response = inference_with_api(image_path, prompt, sys_prompt=system_prompt)
# print(response)

```

    input:
     <|im_start|>system
    As an AI assistant, you specialize in accurate image object detection, delivering coordinates in plain text format 'x1,y1,x2,y2 object'.<|im_end|>
    <|im_start|>user
    find all cakes<|vision_start|><|image_pad|><|vision_end|><|im_end|>
    <|im_start|>assistant
    
    output:
     43,378,996,957 cakes
    



```python

```
