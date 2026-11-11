"""
LlamaIndex 查询路由（Query Routing）完整可运行示例
================================================================

依赖（仓库 .venv 里都已具备）：
    llama-index-core          0.12.x
    llama-index-llms-deepseek 0.1.x
    llama-index-embeddings-huggingface 0.5.x
    llama-index-program-openai        0.3.x   ← PydanticSingleSelector 的隐式依赖
    sqlalchemy / pandas

运行：
    DEEPSEEK_API_KEY=sk-xxx python llamaindex_router_complete.py
    没有 key 也能跑，会自动用 MockLLM 只验证「接线是否正确」。

说明：本文件所有路径都是相对当前工作目录的，SQLite 库文件也会建在 ./
"""

import os
import inspect

from dotenv import load_dotenv

# =============================================================================
# 第 0 步：环境变量（必须在 import llama_index 之前设置）
# =============================================================================
# HF_ENDPOINT 指向镜像站。国内直连 huggingface.co 会超时，
# 而 embedding 模型是要从 HF 下载的，所以这行必须放在最前面。
load_dotenv(verbose=True, override=True)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# =============================================================================
# 第 1 步：导入
# =============================================================================
import pandas as pd
from sqlalchemy import create_engine

from llama_index.core import (
    Settings,           # 全局配置单例：llm / embed_model / node_parser 都从这里取
    VectorStoreIndex,   # 向量索引，用来做「文档问答」引擎
    SQLDatabase,        # SQLAlchemy Engine 的包装器，NLSQL 引擎要的是它而不是裸 engine
    Document,           # 直接构造内存里的文档，省去落盘
)
from llama_index.core.query_engine import RouterQueryEngine  # 本示例主角：路由容器
from llama_index.core.selectors import (
    EmbeddingSingleSelector,      # 纯嵌入相似度选择器
    PydanticSingleSelector,       # 结构化输出选择器（走 function calling）
)
from llama_index.core.schema import QueryBundle  # 构造 select() 需要的查询对象
from llama_index.core.tools import QueryEngineTool       # 把 QueryEngine 包装成「工具」
from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.core.llms import MockLLM

from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# =============================================================================
# 第 2 步：全局配置 Settings
# =============================================================================
# Settings 是 LlamaIndex 的全局单例。所有组件（selector / summarizer / retriever）
# 在 from_defaults() 时都会回落到 Settings 里取 llm 和 embed_model。
# 所以先把这两项配好，后面大量地方就不需要重复传参。
#
# 模型缓存目录：默认在 ~/Library/Caches/llama_index。
# 若遇到 PermissionError（只读文件系统、沙箱环境等），用环境变量指到一个可写目录：
#   LLAMA_INDEX_CACHE_DIR=/tmp/li_cache python llamaindex_router_complete.py
LI_CACHE = os.getenv("LLAMA_INDEX_CACHE_DIR", os.path.expanduser("~/Library/Caches/llama_index"))
os.makedirs(LI_CACHE, exist_ok=True)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-zh-v1.5",  # 中文小模型，约 95MB；换成 bge-base-zh-v1.5 效果更好
    cache_folder=LI_CACHE,
)

# 有 key 用真模型，没 key 用 MockLLM——这样脚本在任何环境都能跑通「构建」这一段
if os.getenv("DEEPSEEK_API_KEY"):
    Settings.llm = DeepSeek(model="deepseek-chat", temperature=0)
else:
    Settings.llm = MockLLM(max_tokens=256)
    print("[提示] 未检测到 DEEPSEEK_API_KEY，使用 MockLLM，只验证接线，不产生真实回答\n")


# =============================================================================
# 第 3 步：准备两类数据源
# =============================================================================
print("=" * 70)
print("第 3 步：构造数据源")
print("=" * 70)

# --- 3.1 文档库：给「向量检索」引擎用 -----------------------------------------
# 直接构造 Document 对象，避免依赖磁盘上的文件夹。
# 真实项目里通常用 SimpleDirectoryReader("./docs").load_data()
docs = [
    Document(text="RAG（检索增强生成）先在知识库中检索相关片段，再把片段拼进 prompt 交给大模型生成答案。"),
    Document(text="文本分块策略常用固定长度分块、递归字符分块和语义分块。递归分块优先按段落切，再按句子切。"),
    Document(text="向量数据库用于存储高维向量并做近似最近邻检索，Milvus、FAISS、Qdrant 都是常见选择。"),
    Document(text="查询改写（Query Rewriting）用于把口语化、有指代的问题改写成适合检索的独立查询。"),
]

# VectorStoreIndex.from_documents 内部会做三件事：
#   1) 用 Settings.node_parser 把 Document 切成分块（Node）
#   2) 用 Settings.embed_model 给每个 Node 算向量
#   3) 建一个内存向量索引
# .as_query_engine() 把它封装成标准 QueryEngine 接口（有 .query(str) 方法）
docs_index = VectorStoreIndex.from_documents(docs)
docs_engine = docs_index.as_query_engine(similarity_top_k=2)


# --- 3.2 业务库：给「Text2SQL」引擎用 ----------------------------------------
# 用 SQLite 建一个最小业务表，模拟真实的订单库。
# 关键点：NLSQLTableQueryEngine 要的是 SQLDatabase 包装器，不是 sqlalchemy Engine。
DB_PATH = "./demo_orders.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)  # 每次重建，保证示例可重复运行

sqlalchemy_engine = create_engine(f"sqlite:///{DB_PATH}")

orders_df = pd.DataFrame(
    {
        "month": ["2025-01", "2025-02", "2025-03", "2025-04"],
        "order_amount": [120000, 98000, 155000, 210000],
        "order_count": [320, 260, 410, 505],
    }
)
# to_sql 会把 DataFrame 落成表；index=False 避免多出一列索引
orders_df.to_sql("orders", sqlalchemy_engine, index=False, if_exists="replace")

# SQLDatabase 是 LlamaIndex 对 SQLAlchemy 的适配层，
# 它会反射表结构，把「表名 + 字段 + 类型」塞进 Text2SQL 的 prompt。
sql_database = SQLDatabase(sqlalchemy_engine)

# tables=["orders"] 显式限定可用表范围。
# 表多时不要全塞进去，schema 会撑爆 prompt——这也是后面要用 ObjectIndex 预筛的原因。
sql_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["orders"],
    llm=Settings.llm,
    verbose=False,
)

print(f"文档引擎: {len(docs)} 篇文档 -> 已建向量索引")
print(f"SQL 引擎: {DB_PATH} -> orders 表 {len(orders_df)} 行")


# =============================================================================
# 第 4 步：把两个引擎包装成 QueryEngineTool
# =============================================================================
print()
print("=" * 70)
print("第 4 步：包装成 Tool（description 是路由的唯一依据）")
print("=" * 70)

# QueryEngineTool.from_defaults 的参数：
#   query_engine   : 被包装的引擎
#   name           : 工具名（注意：这个名字不会进路由 prompt）
#   description    : 工具描述（唯一会进 prompt 的内容，决定路由准确率）
#   return_direct  : 为 True 时，Agent 场景下工具结果直接返回不再过 LLM
tools = [
    QueryEngineTool.from_defaults(
        query_engine=docs_engine,
        name="docs",
        # 好描述 = 说清「适合回答什么」+「边界在哪」，而不是「这是什么」
        description=(
            "适合回答概念解释、原理说明、方法论类问题，"
            "例如什么是 RAG、分块策略怎么选、向量数据库有哪些。"
            "不适合需要精确数字统计的问题。"
        ),
    ),
    QueryEngineTool.from_defaults(
        query_engine=sql_engine,
        name="sql",
        description=(
            "适合回答需要精确统计的业务数据问题，"
            "例如某个月的订单总额、订单量排名、环比变化。"
            "数据源是 orders 表，包含 month / order_amount / order_count 三个字段。"
        ),
    ),
]

for t in tools:
    print(f"  - {t.metadata.name}: {t.metadata.description[:28]}...")

# 小实验：验证「name 不进 prompt、只有 description 进 prompt」
from llama_index.core.selectors.llm_selectors import _build_choices_text

choices_text = _build_choices_text([t.metadata for t in tools])
print("\n  selectors 内部拼出的候选列表（注意只有描述、没有工具名）：")
for line in choices_text.splitlines():
    if line.strip():
        print("    " + line[:70])


# =============================================================================
# 第 5 步：三种路由方式
# =============================================================================

# --- 5.1 默认方式：LLM 意图识别 --------------------------------------------
# from_defaults 内部调用 get_selector_from_llm()，策略是「优先 Pydantic，ValueError 才回落」：
#   try:    PydanticSingleSelector  (function calling / tool_calls)
#   except ValueError: LLMSingleSelector (纯文本解析)
# 只有 ValueError 会被捕获，ImportError（比如没装 llama-index-program-openai）会直接抛出。
router_llm = RouterQueryEngine.from_defaults(query_engine_tools=tools)
print()
print("=" * 70)
print("第 5 步：三种路由方式")
print("=" * 70)
print(f"  ① from_defaults()        -> selector = {type(router_llm._selector).__name__}")

# --- 5.2 多选路由：允许多个引擎一起答，结果由 TreeSummarize 合并 -------------
# select_multi=True 时走 PydanticMultiSelector / LLMMultiSelector，
# 命中多个工具时各引擎并行执行，再用 TreeSummarize 做一次 LLM 汇总（不是字符串拼接）。
router_multi = RouterQueryEngine.from_defaults(query_engine_tools=tools, select_multi=True)
print(f"  ② select_multi=True      -> selector = {type(router_multi._selector).__name__}")

# --- 5.3 纯嵌入相似度路由：不花 token ---------------------------------------
# 机制：把 query 和每个 tool 的 description 各算一个 embedding，
# 取相似度 top1 作为路由结果。零 LLM 调用，成本比 LLM 路由低一到两个数量级。
# 注意：相似度绝对值通常不高（0.3~0.6 很常见），别用阈值卡，只比较相对排序。
router_emb = RouterQueryEngine(
    selector=EmbeddingSingleSelector.from_defaults(),
    query_engine_tools=tools,
)
print(f"  ③ EmbeddingSingleSelector -> selector = {type(router_emb._selector).__name__}")

# --- 5.4 手动指定 PydanticSingleSelector（不想要 from_defaults 的隐式回落时）---
# 这里刻意暴露一个真实的坑：OpenAIPydanticProgram.from_defaults() 内部有一道硬校验
#     if not isinstance(llm, OpenAI): raise ValueError(...)
# 所以只有「OpenAI 兼容类」的 LLM 能用 Pydantic 选择器。
# DeepSeek 的继承链是 DeepSeek -> OpenAILike -> OpenAI，能通过；
# 换成本地 vLLM、Ollama 或 MockLLM 就会 fail，此时必须走 LLMSingleSelector。
router_pyd = None
try:
    router_pyd = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(llm=Settings.llm),
        query_engine_tools=tools,
    )
    print(f"  ④ 手动 PydanticSingle     -> selector = {type(router_pyd._selector).__name__}")
except ValueError as e:
    # 这正是 5.1 里 from_defaults() 会「静默回落」到 LLMSingleSelector 的那个异常
    print(f"  ④ 手动 PydanticSingle     -> 不可用: {e}")
    print("     这解释了为什么 ① 的 from_defaults() 在非 OpenAI 类 LLM 下会回落")


# =============================================================================
# 第 6 步：执行查询并观察路由决策
# =============================================================================
queries = [
    "2025 年 3 月的订单总额是多少",      # 期望走 sql
    "递归字符分块是怎么工作的",           # 期望走 docs
]

# MockLLM 下执行查询只会得到占位答案，但「选哪个引擎」依然是真实的决策结果
print()
print("=" * 70)
print("第 6 步：执行查询，观察路由决策")
print("=" * 70)

for q in queries:
    print(f"\n问题: {q}")

    # --- 6.1 纯嵌入路由：离线也能拿到真实的相似度分数 -----------------------
    emb_result = router_emb._selector.select(
        [t.metadata for t in tools],
        QueryBundle(query_str=q),
    )
    print(f"  [嵌入路由] -> {tools[emb_result.ind].metadata.name}"
          f"   ({emb_result.reason})")

    # --- 6.2 LLM 路由：真实调用大模型 -------------------------------------
    try:
        resp = router_llm.query(q)
    except Exception as e:
        # MockLLM 的输出是占位文本，SelectionOutputParser 可能解析不出编号，属于预期
        print(f"  [LLM 路由] 跳过（{type(e).__name__}: {str(e)[:60]}）")
        continue
    # response.metadata["selector_result"] 就是路由决策记录：
    #   .ind / .index    选中的工具下标
    #   .reason / .reasons  模型给出的选择理由
    sel = resp.metadata.get("selector_result")
    if sel is not None:
        print(f"  [LLM 路由] -> {tools[sel.ind].metadata.name}"
              f"   理由: {sel.reason}")
    print(f"  [回答] {str(resp)[:80]}")


# =============================================================================
# 第 7 步：规模上去之后的两级路由（思路演示，不实际执行）
# =============================================================================
# 工具到几十上百个时，把所有描述塞进 prompt 会超长且路由准确率下降。
# 常规做法是两级：
#   第一级：ObjectIndex.from_objects(tools) 把工具描述建成 VectorStoreIndex，
#           用嵌入检索粗筛出 top-k 候选（便宜、无 token 成本）
#   第二级：把 top-k 候选交给 RouterQueryEngine 做精排（贵、但准确）
#
# 代码骨架：
#   from llama_index.core.objects import ObjectIndex
#   obj_index = ObjectIndex.from_objects(tools)          # 内部就是 VectorStoreIndex
#   top_k_tools = obj_index.as_retriever(similarity_top_k=3).retrieve(query_str)
#   router = RouterQueryEngine.from_defaults(query_engine_tools=top_k_tools)
#
# 这与 RouterRetriever 的区别：RouterRetriever 是作用在 retriever 层的路由，
# 用同一套 selector，但路由对象是 retriever 而不是 query_engine。

print()
print("=" * 70)
print("完成：三种路由方式都已构建成功")
print("=" * 70)

# 顺手打印一下各选择器的来源模块，方便对照源码阅读
src_list = [("EmbeddingSingleSelector", router_emb._selector)]
if router_pyd is not None:
    src_list.insert(0, ("PydanticSingleSelector", router_pyd._selector))
for name, s in src_list:
    print(f"  {name:26s} {inspect.getsourcefile(type(s)).split('site-packages/')[-1]}")