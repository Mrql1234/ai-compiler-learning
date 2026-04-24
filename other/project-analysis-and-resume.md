# AI Robot Agent 项目分析与简历表述

> 分析基于当前仓库代码结构整理，时间点为 2026-04-24。

## 项目定位

基于当前仓库代码，我对项目的判断是：

这是一个“配置驱动”的 AI Robot Agent 服务，核心目标不是做复杂工作流编排，而是让 LLM 直接基于 `System Prompt + 历史消息 + 用户输入` 生成结构化回复；服务层负责配置加载、Prompt 渲染、重试、流式输出、错误处理和监控埋点。

关键入口和实现主要在：

- `app/main.py`
- `app/api/routes/agent.py`
- `app/services/chat_service.py`
- `app/agents/base.py`
- `app/agents/simple_chat_agent.py`
- `app/agents/llm_client.py`

## 项目结构

可以大致理解成下面几层：

```text
cn-ai-robot-agent/
├── app/
│   ├── main.py                 # FastAPI 入口、生命周期、全局异常、信号处理
│   ├── api/
│   │   └── routes/agent.py     # /healthCheck /sync_api /streaming_api
│   ├── services/
│   │   ├── chat_service.py     # 核心编排：加载配置、渲染 prompt、调用 agent、重试、打点
│   │   ├── prompt_renderer.py  # prompt 模板变量替换
│   │   └── errors.py           # LLM 领域错误封装
│   ├── agents/
│   │   ├── base.py             # Agent 抽象接口
│   │   ├── simple_chat_agent.py# 具体 agent：解析 ND-JSON、聚合结果、流式输出
│   │   └── llm_client.py       # 底层 LLM 调用，封装 AgentScope/OpenAIChatModel
│   ├── repositories/
│   │   └── robot_repository.py # 读取 ai_call_robot_agent 配置表
│   ├── schemas/
│   │   ├── chat.py             # 请求/响应/流式事件模型
│   │   └── common.py           # AgentConfig / ErrorOut
│   ├── core/
│   │   ├── config.py           # 配置中心接入
│   │   ├── logging.py          # 日志初始化
│   │   └── middleware.py       # TraceId 中间件
│   └── utils/
│       └── metrics.py          # InfluxDB 指标打点
├── tests/                      # 单测 + 接口验收测试
├── scripts/                    # 测试脚本、端到端验证脚本
├── specs/                      # 需求/设计/任务拆解文档
├── run_local.py                # 本地启动
├── start_api_server.py         # 生产启动
└── pyproject.toml              # 依赖管理
```

## 分层职责

- `main` 层：负责应用启动、配置初始化、日志初始化、生命周期管理、全局异常兜底。
- `api` 层：负责参数校验、HTTP 状态码映射、同步和流式接口暴露。
- `service` 层：负责业务编排，是整个系统的核心 orchestrator。
- `agent` 层：负责和 LLM 交互、解析结构化输出、封装同步/流式能力。
- `repository` 层：负责读取 Agent 配置，连接业务配置和运行时执行逻辑。
- `schema` 层：负责请求、响应、ND-JSON 事件格式标准化。
- `core/utils` 层：负责配置、日志、链路追踪和性能指标。

## Agent 相关设计

这个项目的 agent 设计有几个很明显的特点：

### 1. 不是通用 Agent Framework，而是轻量自定义抽象

- 用 `BaseAgent` 定义统一接口：`run()` 和 `run_stream()`
- 用 `AgentResult` 统一结构化输出：`reply / intent / action / label / buttons`
- 这样业务层只依赖抽象，不直接耦合底层模型 SDK

### 2. 当前只有一个具体实现：`SimpleChatAgent`

- `SimpleChatAgent` 负责：
- 组装消息列表
- 调用 LLM 流式接口
- 解析 ND-JSON
- 把 `reply` 事件归一化成 `text`
- 聚合同步结果，或直接透传流式结果
- 从回复中提取 `<Button>...</Button>` 形成按钮能力

### 3. Agent 的“能力”主要由数据库配置驱动

- agent 配置来自 MySQL 的 `ai_call_robot_agent` 表
- `robot_repository.py` 读取 `model / language / prompt_template / actions / meta_data` 等字段
- `chat_service.py` 根据 `agent_id` 加载配置，再渲染 prompt
- 所以“新增 agent”本质上更像“新增配置”而不是写新代码

### 4. 同步和流式共用一套协议

- 底层统一要求 LLM 产出 ND-JSON 事件流
- 同步接口是“先流式接收，再聚合成完整 `ChatResponse`”
- 流式接口则直接持续输出 ND-JSON
- 这样协议一致，减少了两套输出逻辑分叉

### 5. Service 层才是实际 orchestrator

- `route` 只做参数校验和 HTTP 错误映射
- `chat_service` 负责：
- 加载 agent 配置
- 渲染 prompt
- 调用 agent
- `NoStructuredReplyError` 场景重试
- 记录 TTFT 和总耗时指标
- 所以这里的“agent”更偏向“LLM 执行单元”，真正编排逻辑在 service 层

## 一句话概括 Agent 设计

这是一个“配置驱动 + 统一抽象 + 流式优先”的轻量 Agent 方案：

`BaseAgent` 负责定义边界，`SimpleChatAgent` 负责协议解析与输出聚合，`chat_service` 负责业务编排，数据库配置决定不同 agent 的模型与 prompt 行为。

## 简历概述基础版

你可以直接用这版，偏工程化表达：

1. 负责设计并开发基于 FastAPI 的 LLM 对话服务，采用“System Prompt 直驱”方案替代 DAG 编排，实现同步问答与 ND-JSON 流式输出两套接口。
2. 抽象自定义 Agent 体系，基于 `BaseAgent + SimpleChatAgent` 封装模型调用、结构化事件解析、重试容错与按钮指令提取，支持按数据库配置动态切换模型与 Prompt。
3. 搭建完整服务化能力，包括 MySQL 配置管理、配置中心接入、TraceId 日志链路、InfluxDB 延迟指标监控，以及单元测试和接口验收测试，支撑本地开发与 K8s 部署。

## 更偏简历成就导向的版本

这一版更强调“你做成了什么”和“带来了什么结果”，适合简历项目经历直接落地：

1. 主导落地配置驱动的 AI Robot Agent 服务，基于 FastAPI 构建同步与流式对话接口，支持 LLM 直接按 System Prompt 输出结构化结果，提升机器人迭代效率与多场景复用能力。
2. 设计并实现轻量 Agent 执行架构，统一封装 Prompt 渲染、流式解析、失败重试、结构化动作提取和多模型接入，降低新增机器人和切换模型的开发成本。
3. 完善服务稳定性与可观测性建设，补齐异常兜底、TraceId 链路追踪、TTFT/总耗时指标监控及测试体系，支撑服务在本地开发、测试和 K8s 环境中的稳定运行。

## 更偏面试项目介绍的版本

这一版更适合你在面试里口述，强调“背景、方案、职责、结果”：

1. 这个项目是一个 AI Robot Agent 服务，主要面向业务机器人对话场景。和传统 DAG 编排不同，我们采用的是 System Prompt 直驱方式，让大模型直接输出结构化回复，从而减少流程配置复杂度。
2. 我主要负责服务端核心链路设计和实现，包括 FastAPI 接口、Agent 抽象、Prompt 渲染、数据库配置加载、LLM 流式调用、ND-JSON 协议解析，以及同步和流式两种返回模式的统一封装。
3. 这个项目的一个特点是“配置驱动”，机器人能力主要由数据库中的模型、语言、Prompt 模板等配置决定；另一个特点是“流式优先”，同步接口本质上也是基于流式协议聚合出来的，这样实现更统一，也更方便做监控和容错。

## 更偏架构设计亮点的版本

这一版适合技术面、架构面，突出设计取舍：

1. 采用“接口层薄、服务层编排、Agent 层执行、Repository 层配置读取”的分层架构，将 HTTP 处理、业务编排、模型调用和配置访问解耦，降低代码耦合度并提升可维护性。
2. 设计统一的 ND-JSON 事件协议，把同步响应和流式响应收敛到同一条执行链路，通过 `run_stream()` 作为底座、`run()` 作为聚合视图，减少双实现带来的协议分叉和维护成本。
3. 通过数据库配置驱动 Agent 行为，并在底层封装模型缓存、异常映射、重试机制、链路追踪和时延指标，实现模型层、业务层和运维监控层的清晰边界与可扩展能力。

## 使用建议

- 写简历时，优先使用“更偏简历成就导向”的版本。
- 面试自我介绍或项目介绍时，优先使用“更偏面试项目介绍”的版本。
- 技术复盘、架构答辩或晋升材料里，优先使用“更偏架构设计亮点”的版本。

## 可继续精炼的方向

如果后续还想继续压缩成更强的简历表达，可以再往下面几个方向收敛：

1. 增加业务规模信息，比如日调用量、响应时延、服务实例规模、支撑的机器人数量。
2. 增加结果性指标，比如接口延迟降低、迭代效率提升、Prompt 配置成本下降。
3. 增加个人角色信息，比如主导设计、独立负责、跨团队协作、推动上线。

## 同步和流式的区别

同步和流式，最大的区别在于：结果是一次性返回，还是边生成边返回。

### 核心区别

1. 同步
- 服务端要等模型把完整结果都生成完，再一次性返回给调用方
- 调用方拿到的是最终完整 JSON
- 在这个项目里对应 `POST /sync_api`
- 返回模型是 `app/schemas/chat.py` 里的 `ChatResponse`

2. 流式
- 服务端一边从模型接收内容，一边把内容持续推给调用方
- 调用方会先收到第一段，再陆续收到后续片段
- 在这个项目里对应 `POST /streaming_api`
- 返回的是 `application/x-ndjson` 事件流，事件类型包括 `text`、`intent`、`action`、`label`、`error`

### 直观类比

1. 同步：像点外卖，骑手到了才一次性交付
2. 流式：像直播，你能边看边接收内容

### 在当前项目里的实现区别

从实现上看，这个项目其实是“流式优先”：

- 流式底座在 `app/agents/simple_chat_agent.py` 的 `run_stream()`
- 同步是在 `app/agents/simple_chat_agent.py` 的 `run()` 里，把流式 ND-JSON 事件聚合成一个完整结果
- 编排逻辑在 `app/services/chat_service.py`

也就是说：

1. 流式接口：直接把每一行 ND-JSON 往外发
2. 同步接口：先内部走一遍流式，再把所有 `text/intent/action/label` 聚合成一个最终响应

### 各自适合的场景

1. 适合同步的场景
- 后端系统之间调用，调用方只关心最终结果
- 业务逻辑必须拿到完整答案后才能继续处理
- 对接传统 HTTP/JSON 系统更方便
- 例如：
- 机器人问答后，直接把完整回复入库
- 需要拿完整 `intent/action/label` 再做后续业务判断
- 批处理、定时任务、内部服务调用

2. 适合流式的场景
- 用户在前端聊天界面里，希望立刻看到模型开始回复
- 回复比较长，不能让用户一直等完整结果
- 需要更好的实时体验和更低体感延迟
- 例如：
- Web 聊天窗口
- App 客服对话页
- 实时字幕、实时建议生成
- 长文本生成、长答案问答

### 各自优缺点

1. 同步优点
- 接口简单，调用方好处理
- 天然适合标准 JSON API
- 更容易做事务型业务

2. 同步缺点
- 用户必须等完整结果
- 长回复时体验差
- 很难展示中间过程

3. 流式优点
- 首包快，体感更好
- 适合长文本生成
- 更适合聊天型产品

4. 流式缺点
- 客户端处理更复杂
- 错误处理更麻烦
- 需要约定事件协议，比如 ND-JSON

### 一句话总结

- 同步：要最终结果，适合系统调用和结构化处理
- 流式：要实时体验，适合聊天界面和长文本生成

## ND-JSON 协议在项目里的用法

在这个项目里，ND-JSON 可以理解成：模型和服务之间约定的一种“逐行输出结构化事件”的协议。

它的作用不是存数据文件，而是作为 LLM 输出协议来统一同步和流式两种接口。

### 这个项目里的 ND-JSON 是什么

ND-JSON 的特点是：

- 每一行都是一个独立 JSON 对象
- 行与行之间用换行符 `\n` 分隔
- 可以边生成边传输，不需要等完整大 JSON 一次性拼完

例如模型可能输出：

```json
{"type":"text","delta":"你好"}
{"type":"text","delta":"，请问有什么可以帮你？"}
{"type":"intent","name":"consult"}
{"type":"label","name":"normal"}
```

在这个项目里，服务端就是按这种“一行一个事件”的方式解析和转发。

### 为什么项目里要用 ND-JSON

因为这个项目既要支持流式输出，也要支持同步聚合。

如果直接让模型输出一个完整 JSON，会有几个问题：

- 流式场景下很难边生成边解析
- 半截 JSON 很容易解析失败
- 同步和流式要维护两套协议

所以这里选了 ND-JSON，原因很实际：

- 每行独立，天然适合流式
- 即使某一行坏了，也不一定影响整次请求
- 可以把同步和流式统一到同一条执行链路

### 项目里的事件类型

事件模型定义在 `app/schemas/chat.py`：

1. `text`
- 文本增量
- 字段是 `delta`

2. `intent`
- 意图识别结果
- 字段是 `name`

3. `action`
- 动作指令
- 字段是 `name` 和 `action_params`

4. `label`
- 标签分类结果
- 字段是 `name`

5. `error`
- 错误事件
- 字段是 `message`

### 项目里如何解析 ND-JSON

核心逻辑在 `app/agents/simple_chat_agent.py` 的 `run_stream()`。

处理方式大致是：

1. 从底层 LLM 流式接口不断拿到文本片段
2. 先把片段累积到 `line_buffer`
3. 只要发现有换行符，就切出完整的一行
4. 对这一行做 `json.loads()`
5. 如果不是合法 JSON，或者没有 `type`，就跳过并记日志
6. 如果 `type == "reply"`，会标准化成 `text`
7. 最后把合法事件重新按 `line + "\n"` 输出

也就是说，这里不是按大块 JSON 解析，而是按一行一个事件解析。

### `reply` 为什么会被转成 `text`

在 `app/agents/simple_chat_agent.py` 里有一层兼容处理：

- 如果模型返回的是 `{"type":"reply", ...}`
- 服务端会把它统一改成 `{"type":"text", ...}`

这是为了统一协议，避免上下游同时处理 `reply` 和 `text` 两种文本事件名。

### ND-JSON 在流式接口里的用法

流式接口在 `app/api/routes/agent.py` 的 `/streaming_api`。

这里会：

1. 先校验参数
2. 先检查 agent 是否存在
3. 调用 `app/services/chat_service.py` 的 `handle_chat_stream()`
4. `handle_chat_stream()` 再调用 `SimpleChatAgent.run_stream()`
5. 最终通过 `StreamingResponse` 持续返回 `application/x-ndjson`

所以流式接口本质上就是：

- 模型输出 ND-JSON
- 服务端校验、标准化
- 原样一行一行往客户端推

### ND-JSON 在同步接口里的用法

同步接口 `/sync_api` 并不是单独搞了一套完全不同的协议，而是：

1. 内部仍然调用 agent 的流式逻辑
2. 一边消费 ND-JSON 事件
3. 一边把事件聚合成最终结果

聚合逻辑在 `app/agents/simple_chat_agent.py` 的 `run()`：

- `text` 事件追加到 `reply_parts`
- `intent` 收集到 `intents`
- `action` 收集到 `actions`
- `label` 收集到 `labels`

最后再拼成 `AgentResult`，由 `app/services/chat_service.py` 转成 `ChatResponse` 返回。

可以理解成：

- 流式接口：直接输出 ND-JSON
- 同步接口：消费 ND-JSON，再组装成完整 JSON

### ND-JSON 在这个项目里解决了什么问题

它主要解决了 4 件事：

1. 统一同步和流式协议
- 两种接口共用一套事件模型

2. 增强容错
- 某一行坏了可以跳过，不一定导致整段失败

3. 适配 LLM 流式输出
- 模型本来就是分片返回，按行切事件更自然

4. 支持结构化多维输出
- 不只是文本，还可以输出 `intent`、`action`、`label`

### 一句话总结

这个项目里的 ND-JSON，本质上是 LLM 到服务、服务到客户端之间的结构化事件协议：它让模型输出可以边生成边传输、逐行校验、统一聚合，同时把同步接口和流式接口收敛到同一套实现上。

## 一般链路有哪些节点

如果说“一个请求从进来到返回”的一般链路节点，在这个项目里大致可以拆成下面这些：

### 1. 接入层

- 客户端发起请求
- 进入 FastAPI 路由
- 对应接口是：
- `POST /sync_api`
- `POST /streaming_api`

这一层主要做：

- 参数接收
- 基础校验
- HTTP 状态码映射

### 2. 中间件层

- 请求先经过中间件
- 这里主要是 TraceId 注入
- 代码在 `app/core/middleware.py`

这一层主要做：

- 提取或生成 `X-Trace-Id`
- 让后续日志都能串起来

### 3. Service 编排层

- 路由把请求交给 service
- 核心在 `app/services/chat_service.py`

这一层主要做：

- 加载 agent 配置
- 渲染 system prompt
- 调用 agent
- 失败重试
- 记录时延指标
- 组织最终返回结果

这层可以理解成整个链路的调度中心。

### 4. 配置和数据读取层

- service 会根据 `agent_id` 去查数据库配置
- 代码在 `app/repositories/robot_repository.py`

这一层主要做：

- 读取 `ai_call_robot_agent` 表
- 拿到模型名、语言、Prompt 模板等配置
- 给后续 agent 执行提供运行参数

### 5. Prompt 处理层

- 拿到配置后，需要把模板渲染成最终 system prompt
- 代码在 `app/services/prompt_renderer.py`

这一层主要做：

- 替换 `{language}`
- 替换 `{meta_data}`
- 生成最终发给 LLM 的 system prompt

### 6. Agent 执行层

- service 调用 agent 抽象
- 抽象定义在 `app/agents/base.py`
- 具体实现是 `app/agents/simple_chat_agent.py`

这一层主要做：

- 组装历史消息和用户消息
- 调用底层 LLM
- 接收流式返回
- 解析 ND-JSON 事件
- 聚合成 `reply / intent / action / label / buttons`

### 7. LLM 调用层

- agent 再往下调用模型客户端
- 代码在 `app/agents/llm_client.py`

这一层主要做：

- 创建和缓存模型实例
- 调用 AgentScope 的 `OpenAIChatModel`
- 处理流式增量文本
- 把 SDK 异常转换成业务可理解的错误

### 8. 协议解析层

- 这是这个项目比较特别的一个节点
- 模型输出不是直接纯文本，而是 ND-JSON 事件流
- 解析也在 `app/agents/simple_chat_agent.py`

这一层主要做：

- 按行切分
- `json.loads`
- 校验是否有 `type`
- 兼容 `reply -> text`
- 跳过非法行
- 提取结构化事件

### 9. 响应组装层

根据接口不同会分两种：

1. 同步链路
- 聚合所有事件
- 组装成 `ChatResponse`
- 一次性返回 JSON

2. 流式链路
- 事件一条条往外推
- 返回 `application/x-ndjson`

对应代码在：

- `app/services/chat_service.py`
- `app/api/routes/agent.py`

### 10. 监控与日志层

这类节点不会单独处理业务，但贯穿整条链路：

- 日志初始化：`app/core/logging.py`
- 指标打点：`app/utils/metrics.py`
- 应用生命周期和全局异常：`app/main.py`

主要做：

- TraceId 日志串联
- TTFT、总耗时打点
- 全局异常捕获
- 进程信号和 asyncio 异常记录

### 如果抽象成一条完整链路

可以简单记成：

```text
客户端
-> 路由/API
-> 中间件
-> Service 编排
-> Repository 读配置
-> Prompt 渲染
-> Agent 执行
-> LLM Client
-> 模型服务
-> ND-JSON 解析
-> 同步聚合 / 流式透传
-> 返回客户端
```

### 如果在面试里回答

可以讲成 3 层重点：

1. 入口层
- API + Middleware，负责接入、校验、TraceId

2. 核心编排层
- Service + Repository + Prompt Renderer，负责加载配置和组织执行流程

3. 模型执行层
- Agent + LLM Client，负责模型调用、ND-JSON 解析、同步聚合和流式输出

### 一句话总结

一般链路节点可以概括为：接入、校验、编排、配置读取、Prompt 构造、模型调用、协议解析、结果组装、监控日志。
