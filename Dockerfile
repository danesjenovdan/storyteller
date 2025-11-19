FROM python:3.13

# RUN /usr/local/bin/python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y ffmpeg

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY . /app

# RUN python3 manage.py compilemessages

EXPOSE 8000

CMD exec python manage.py runserver 0.0.0.0:8000
