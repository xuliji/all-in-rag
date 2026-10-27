# 本示例演示：把自然语言问题转换成“语义查询 + 元数据过滤条件”，再交给 Chroma 检索。
# 执行流程：
#   B 站视频链接 → Document 文档 → 整理 metadata → Ollama 生成向量 → 存入 Chroma
#   用户问题 → DeepSeek 生成结构化查询 → ChromaTranslator 转换条件 → 检索文档并展示
#
# 两类模型的职责不同：
#   Ollama 嵌入模型：把文本转换成数值向量，用于计算语义相似度。
#   DeepSeek 大语言模型：理解问题并生成查询条件，这里不负责撰写最终回答。
#
# 运行前需要启动 Ollama，并准备 qwen3-embedding:latest 模型，以及有效的 DeepSeek API Key。
# 当前示例的边界：自查询支持过滤和限制返回数量，不能自动完成“按时长排序”。

# os 用于读取模型名称、服务地址和 API Key 等环境变量。
import os

# 从 .env 文件读取配置，避免把 API Key 直接写进代码。
from dotenv import load_dotenv
# DeepSeek 的 LangChain 接口，供自查询检索器调用大语言模型。
from langchain_deepseek import ChatDeepSeek 
# 把 B 站视频信息封装成 LangChain 的 Document 对象。
from langchain_community.document_loaders import BiliBiliLoader
# LangChain 1.x 已将这些传统链和检索器组件移到 langchain_classic。
# AttributeInfo 描述可供大语言模型生成过滤条件的元数据字段。
from langchain_classic.chains.query_constructor.base import AttributeInfo
# SelfQueryRetriever 负责串联“理解问题、构造过滤条件、查询向量库”。
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
# 将 LangChain 的结构化过滤表达式转换为 Chroma 接受的查询参数。
from langchain_community.query_constructors.chroma import ChromaTranslator
# Chroma 保存文档、向量和元数据，并支持相似度检索与元数据过滤。
from langchain_community.vectorstores import Chroma
# 通过 Ollama 的 HTTP 接口调用本机嵌入模型。
from langchain_ollama import OllamaEmbeddings
# 输出 HTTP 请求和检索过程等日志，便于观察执行到哪一步。
import logging

# INFO 级别会显示模型请求和生成的结构化查询；弃用警告属于依赖库提示。
logging.basicConfig(level=logging.INFO)
# 加载找到的 .env 文件；override=True 表示文件中的值会覆盖进程已有的同名环境变量。
# 后面的 os.getenv(...) 因此可以读取 DEEPSEEK_API_KEY 等配置。
load_dotenv(override=True)

# 1. 初始化视频数据
# 用三个固定视频构建演示数据集，后续检索只在这些视频中进行。
video_urls = [
    "https://www.bilibili.com/video/BV1Bo4y1A7FU", 
    "https://www.bilibili.com/video/BV1ug4y157xA",
    "https://www.bilibili.com/video/BV1yh411V7ge",
]

# bili 用于保存整理后的 Document，供向量库批量写入。
bili = []
try:
    # 此处只配置加载器，真正的网络请求发生在下面的 load() 中。
    # 未传入登录凭据时，当前 BiliBiliLoader 只获取视频信息，不获取字幕，
    # 返回的 Document.page_content 为空字符串，视频信息仍保存在 metadata 中。
    loader = BiliBiliLoader(video_urls=video_urls)
    docs = loader.load()
    
    # 一个视频对应一个 Document：page_content 存文本，metadata 存字段信息。
    for doc in docs:
        # B 站原始返回值含有 owner、stat 等嵌套字典，先保留引用以便提取字段。
        original = doc.metadata
        
        # 把嵌套字段整理成 Chroma 可直接保存和过滤的扁平字段。
        # title、author、source 为字符串；view_count、length 为数值。
        # dict.get(键, 默认值) 在缺少字段时提供兜底值，避免直接索引引发 KeyError。
        # duration 的单位是秒，因此后面“时长大于600秒”对应 length > 600。
        # 注意：缺失的数值被记成 0，这只是当前示例的默认值，不代表真实统计值。
        metadata = {
            'title': original.get('title', '未知标题'),
            'author': original.get('owner', {}).get('name', '未知作者'),
            'source': original.get('bvid', '未知ID'),
            'view_count': original.get('stat', {}).get('view', 0),
            'length': original.get('duration', 0),
        }
        
        # 替换原始元数据，仅保留上面选出的字段；page_content 不会在这里被改写。
        # source 保存视频 BV 号以便追踪来源，但未在后面声明为可查询的过滤字段。
        doc.metadata = metadata
        bili.append(doc)
        
except Exception as e:
    # 捕获本段数据加载或整理过程中的异常，并显示原因。
    # try 覆盖整个批次；若 loader.load() 中途失败，不会逐个跳过失败视频重试。
    print(f"加载BiliBili视频失败: {str(e)}")

# 没有文档时提前退出，避免向空数据集建库或继续请求大语言模型。
if not bili:
    print("没有成功加载任何视频，程序退出")
    exit()

# 2. 创建向量存储
# 初始化嵌入客户端；真正生成向量的请求会在写入文档和执行相似度检索时发生。
embed_model = OllamaEmbeddings(
    # 使用环境变量指定模型；未设置时使用本机已安装的 qwen3-embedding:latest。
    model=os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:latest"),
    # Ollama 服务地址，默认连接本机 11434 端口。
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    # 直接连接本机 Ollama，避免 HTTPX 使用系统代理后返回 503。
    client_kwargs={"trust_env": False},
)
# from_documents 会提取每个 Document 的 page_content，调用 embed_documents 生成向量，
# 再把文本、向量和 metadata 一起写入 Chroma。查询时使用同一个嵌入模型生成查询向量。
# metadata 不会自动拼接到 page_content 中参与嵌入；当前无字幕模式下，嵌入的是空文本，
# 因而这里主要演示元数据过滤，不能据此获得有效的视频内容语义检索效果。
# 此处没有指定 persist_directory，不把该演示集合配置为磁盘持久化存储。
vectorstore = Chroma.from_documents(bili, embed_model)

# 3. 配置元数据字段信息
# 这些描述是给 DeepSeek 理解数据结构用的，不会修改已经写入 Chroma 的文档。
# name 必须与 doc.metadata 中的键一致；type 说明过滤值应是什么类型；
# description 解释字段含义，帮助模型把“时长”“观看次数”等用语映射到具体字段。
metadata_field_info = [
    AttributeInfo(
        name="title",
        description="视频标题（字符串）",
        type="string", 
    ),
    AttributeInfo(
        name="author",
        description="视频作者（字符串）",
        type="string",
    ),
    AttributeInfo(
        name="view_count",
        description="视频观看次数（整数）",
        type="integer",
    ),
    AttributeInfo(
        name="length",
        description="视频长度（整数）",
        type="integer"
    )
]

# 4. 创建自查询检索器
# 创建 DeepSeek 客户端；此处尚未调用模型，调用发生在 retriever.invoke(query) 时。
llm = ChatDeepSeek(
    # 使用用户配置的 DeepSeek 模型，通过它把自然语言问题转换成结构化查询。
    model="deepseek-flash",
    # 降低输出随机性，便于生成较稳定的查询条件；不保证每次输出完全一致。
    temperature=0, 
    # 从前面加载的环境变量中读取密钥，不在代码或日志中直接展示密钥。
    api_key=os.getenv("DEEPSEEK_API_KEY")
    )

# from_llm 会根据文档描述、字段信息等构造提示词和输出解析器，组装查询处理流程。
# 每次查询时，模型输出会被解析为 StructuredQuery，主要包含：
#   query：用于向量相似度检索的文本。
#   filter：对 metadata 的约束，例如 gt("length", 600)。
#   limit：可选的结果数量上限。
retriever = SelfQueryRetriever.from_llm(
    # 负责理解问题并生成结构化查询的大语言模型。
    llm=llm,
    # 负责保存数据和执行实际检索的向量库。
    vectorstore=vectorstore,
    # 用自然语言告诉模型：这批文档记录的是什么内容。
    document_contents="记录视频标题、作者、观看次数等信息的视频元数据",
    # 提供允许查询的字段名称、字段类型和含义。
    metadata_field_info=metadata_field_info,
    # 显式选择 Chroma，避免自动检测时导入无关的向量库组件。
    # 例如把 gt("length", 600) 转为 Chroma 的 filter={"length": {"$gt": 600}}。
    structured_query_translator=ChromaTranslator(),
    # 允许模型提取“返回两个”等数量要求，并将 limit 转为检索参数 k。
    # 这个开关只控制返回数量，不会把“最短”转换成按 length 升序排序。
    enable_limit=True,
    # 配合前面的 INFO 日志，输出模型生成的结构化查询，便于检查条件是否正确。
    verbose=True
)

# 5. 执行查询示例
# 第一个问题涉及排序：当前检索器没有排序能力，返回一条结果不等于找到了最短视频。
# 若要准确回答它，需要取出候选视频，再按 metadata["length"] 排序或取最小值。
# 第二个问题涉及过滤：它适合本例，预期生成 length > 600 的元数据约束。
queries = [
    "时间最短的视频",
    "时长大于600秒的视频"
]

for query in queries:
    print(f"\n--- 查询: '{query}' ---")
    # invoke 会依次执行：
    #   1. 把问题和字段描述等信息发送给 DeepSeek。
    #   2. 解析模型返回的 query、filter 和可选 limit。
    #   3. 用 ChromaTranslator 将过滤表达式转换成向量库参数。
    #   4. 用 Ollama 将语义查询文本嵌入为向量，交给 Chroma 结合过滤条件进行检索。
    # 返回值是 list[Document]，而不是大语言模型撰写的自然语言回答。
    # 这里未捕获模型或检索异常；失败时会保留完整 traceback，方便定位问题。
    results = retriever.invoke(query)
    if results:
        # 展示命中文档的 metadata；没有输出字幕正文或相似度分数。
        # 结果顺序取决于相似度检索，不代表已经按时长或观看次数排序。
        for doc in results:
            # 读取展示字段；如果字段缺失，使用提示文字作为兜底。
            title = doc.metadata.get('title', '未知标题')
            author = doc.metadata.get('author', '未知作者')
            view_count = doc.metadata.get('view_count', '未知')
            length = doc.metadata.get('length', '未知')
            print(f"标题: {title}")
            print(f"作者: {author}")
            print(f"观看次数: {view_count}")
            print(f"时长: {length}秒")
            print("="*50)
    else:
        # 查询成功但没有命中文档时，说明当前数据集未返回满足该查询的结果。
        print("未找到匹配的视频")
