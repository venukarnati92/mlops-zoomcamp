FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "Pipfile", "./" ]

RUN pipenv lock

RUN pipenv install --system --deploy

#copy the python batch.py file
COPY batch.py /app/batch.py

#execute the batch.py file with arguments year and month
ENTRYPOINT ["python", "/app/batch.py"]

# Example usage:
# docker build -t batch-job .
# docker run --rm batch-job --year 2023 --month 5