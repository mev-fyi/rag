FROM python:3.9

WORKDIR /app

RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY src/Llama_index_sandbox /app
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r /app/requirements.txt

ENV FLASK_APP=app.py

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
