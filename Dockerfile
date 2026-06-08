# Base image: Python 3.11 on a minimal Debian Linux
# "slim" means a small base image — no extras like docs or development tools
FROM python:3.11-slim

# Set working directory inside the container
# All subsequent commands run from /app
WORKDIR /app

# Install system dependencies that Python packages may need
# - build-essential: required by some pip packages (xgboost, lightgbm) to compile C code
# - libgomp1: OpenMP runtime, needed by XGBoost and LightGBM for parallelism
# - Cleanup at the end keeps the image small
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first, separately from the rest of the code
# This is a Docker optimization: when only your code changes (not deps),
# Docker can reuse the cached "pip install" layer instead of reinstalling
COPY requirements.txt .

# Upgrade pip and install Python dependencies
# --no-cache-dir keeps the image smaller by not storing pip's download cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
# Each folder is copied with the same name on the container
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
COPY setup.py .

# Install the local 'src' package in editable mode
# Same setup.py trick we used during development — makes 'from src.x import y' work
RUN pip install --no-cache-dir -e .

# Tell Docker that this container listens on port 8501
# This is informational — doesn't actually open the port; the runtime does
EXPOSE 8501

# Set Streamlit-specific environment variables
# These match the CMD below and silence the "first-time setup" prompt
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Healthcheck: lets Docker/orchestrators know if the app is alive
# Streamlit exposes /_stcore/health which returns 200 OK when healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# The command that runs when the container starts
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]