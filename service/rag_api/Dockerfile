FROM python:3.11-slim

# 로케일 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
 && rm -rf /var/lib/apt/lists/*

# 한국어 로케일 생성
RUN sed -i 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen

RUN adduser --disabled-password --gecos "" appuser

USER appuser

# Python / Locale 환경 변수
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=ko_KR.UTF-8 \
    LANGUAGE=ko_KR:ko:en \
    LC_ALL=ko_KR.UTF-8

WORKDIR /app

COPY service/rag_api/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY core ./core 
COPY service/rag_api/ .

EXPOSE 8000
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]



