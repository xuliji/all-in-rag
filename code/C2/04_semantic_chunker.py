from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="embeddinggemma-2:latest",
)

# 初始化 SemanticChunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile" # 也可以是 "standard_deviation", "interquartile", "gradient"
)

loader = TextLoader("../../data/C2/txt/蜂医.txt", encoding="utf-8")
documents = loader.load()

docs = text_splitter.split_documents(documents)

print(f"文本被切分为 {len(docs)} 个块。\n")
print("--- 前2个块内容示例 ---")
for i, chunk in enumerate(docs[:2]):
    print("=" * 60)
    print(f'块 {i+1} (长度: {len(chunk.page_content)}):\n"{chunk.page_content}"')
