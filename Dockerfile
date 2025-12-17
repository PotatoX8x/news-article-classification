FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

EXPOSE 5000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]
