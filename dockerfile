# 使用官方轻量版Python镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# ✅ 安装系统依赖
RUN apt-get update && apt-get install -y gcc

# 拷贝依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝本地代码到容器
COPY . .

# 设置容器启动命令
CMD ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
