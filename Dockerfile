FROM python:3.9.2

# upgrade pip
# RUN python -m pip install --upgrade pip

# get curl for healthchecks
# RUN apk add curl

# permissions and nonroot user for tightened security
# RUN adduser -D nonroot
# RUN mkdir /home/app/ && chown -R nonroot:nonroot /home/app
# RUN mkdir -p /var/log/flask-app && touch /var/log/flask-app/flask-app.err.log && touch /var/log/flask-app/flask-app.out.log
# RUN chown -R nonroot:nonroot /var/log/flask-app

# USER nonroot

# # copy all the files to the container
# COPY --chown=nonroot:nonroot . .
WORKDIR /home/app
COPY . .

# venv
# ENV VIRTUAL_ENV=/home/app/venv

# python setup
# RUN python -m venv $VIRTUAL_ENV
# ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r requirements.txt
RUN export FLASK_APP=app.py

# define the port number the container should expose
EXPOSE 6000

# CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
CMD ["python", "app.py"]
# CMD ["flask", "run"]