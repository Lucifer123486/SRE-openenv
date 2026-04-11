# 1. Use a standard Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy everything into the /app directory
COPY . .

# 4. Install all dependencies including 'openai' for the Proxy check
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    requests \
    openai \
    openenv-core>=0.2.0 && \
    pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# 5. Expose Port 7860 (Standard for Hugging Face & OpenEnv)
EXPOSE 7860

# 6. Run the app using the new folder structure
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]