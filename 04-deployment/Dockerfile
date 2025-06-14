FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# do stuff here
RUN pip install --no-cache-dir \
    'pandas' \
    'scikit-learn==1.6.1'

#copy the python batch.py file
COPY batch.py /app/batch.py

#execute the batch.py file with arguments year and month
ENTRYPOINT ["python", "/app/batch.py"]
CMD ["2023", "10"]
# Example usage:
# docker build -t batch-job .
# docker run --rm batch-job 2023 10