# ============================================================
# AI 旅行 Agent - 全局配置
# 支持 .env 文件 + 环境变量覆盖
# ============================================================
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env
ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# ---------- 基础路径 ----------
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"
VECTOR_DIR = STORAGE_DIR / "vector_index"
KB_FILE = DATA_DIR / "travel_cases.json"


# ---------- LLM ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
# 未配置 API Key 时自动使用演示模式(mock LLM),保证全流程可离线跑通
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # auto | deepseek | mock

# ---------- Embedding ----------
# auto: 优先 sentence-transformers(本地 bge-small-zh-v1.5),否则回退内置 local_hash
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "auto")  # auto | sentence-transformers | local_hash
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")  # 可选: DashScope text-embedding-v3

# ---------- RAG ----------
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.22"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "350"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "40"))

# ---------- 高德地图(Web 服务 API)----------
AMAP_KEY = os.getenv("AMAP_KEY", "")
AMAP_BASE_URL = os.getenv("AMAP_BASE_URL", "https://restapi.amap.com")
