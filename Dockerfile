# 1. Use a standard Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy everything into the /app directory
COPY . .

# 4. Install all required dependencies (Added openenv-core as required by the error)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    requests \
    openenv-core>=0.2.0

# 5. Expose Port 7860 (Standard for Hugging Face & OpenEnv)
EXPOSE 7860

# 6. Run the app from the new 'server' folder location
# We use server.app:app because main.py was renamed to app.py inside the server/ folder
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]