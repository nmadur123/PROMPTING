FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Kutubxonalar alohida qatlamda — kod o'zgarganda qayta o'rnatilmaydi.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY data ./data

# ML artifaktlari image ichida bo'lmasa, birinchi so'rovda o'zi quriladi.
# Bu yerda oldindan quramiz — konteyner "sovuq start"siz ko'tariladi.
RUN python -c "from app.ml.engine import engine; engine.warmup(); print(engine.stats())"

EXPOSE 8010
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
