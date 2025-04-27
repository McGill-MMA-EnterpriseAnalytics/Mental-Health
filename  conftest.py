# conftest.py

import sys
import os

# 获取当前 conftest.py 所在目录（就是 sample/）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 把 sample/ 加入 sys.path（如果还没加）
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
