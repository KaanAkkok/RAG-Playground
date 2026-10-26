FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng \
        ffmpeg \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
        dvisvgm \
        cm-super \
        libgl1 libglib2.0-0 \
        build-essential libcairo2-dev libpango1.0-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

COPY pyproject.toml uv.lock README.md ./
COPY manimations ./manimations
RUN uv sync --frozen --no-install-project --group notebook

COPY src ./src
COPY scripts ./scripts
COPY notebooks ./notebooks
COPY data/animations ./data/animations
COPY data/evaluation ./data/evaluation
RUN uv sync --frozen --all-packages --group notebook --no-editable
RUN mkdir -p data/documents/raw_documents data/documents/saved_documents data/evaluation

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.root_dir=/app", "--ServerApp.default_url=/lab/tree/notebooks", "--ServerApp.token=", "--ServerApp.password="]
