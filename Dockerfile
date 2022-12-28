FROM python:3.9.10

WORKDIR /app

COPY . .

RUN pip3 install --upgrade pip & \
    pip3 --no-cache-dir install -r requirements.txt


EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["src/app.py"]