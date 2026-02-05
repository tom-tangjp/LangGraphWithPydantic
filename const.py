from pathlib import Path
from utils import env

DEFAULT_ROOT = Path(__file__).resolve().parent

OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://localhost:11434")

TRUNCATE_LOG_LEN = 8192

### Web Request Config
WEB_MAX_CHARS = 12000
WEB_TIMEOUT_S = 15

WEB_SEARCH_MAX_RESULTS = 8
WEB_SEARCH_RECENCY_DAYS = 30


MAX_STEP_RETRIES = 5  # 单 step 最大 retry 次数
MIN_OUTPUT_CHARS = 40  # 输出太短也视为“没产出/退化”，可按需调

MAX_ITERATIONS = 20  # 最大迭代次数
MAX_TOOL_CONSECUTIVE_COUNT = 5  # 最大连续调用工具次数
