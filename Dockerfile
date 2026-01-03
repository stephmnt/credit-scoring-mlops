FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY app_entry.py app.py gradio_app.py ./
COPY src/ src/
COPY data/ data/
COPY artifacts/ artifacts/

EXPOSE 7860

CMD ["uvicorn", "app_entry:app", "--host", "0.0.0.0", "--port", "7860"]
