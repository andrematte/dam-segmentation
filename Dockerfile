FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.4.2

# ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY dam_segmentation ./dam_segmentation
COPY dist/ms_image_tool-0.1.0-py3-none-any.whl ./dist/ms_image_tool-0.1.0-py3-none-any.whl

COPY README.md ./

RUN poetry install --without dev --no-interaction --no-ansi

COPY scripts ./scripts

# ENTRYPOINT ["python", "scripts/create_sample_data.py"]