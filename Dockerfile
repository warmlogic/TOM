FROM python:3.7.4-slim

WORKDIR /app

ADD . /app

RUN pip install --trusted-host pypi.python.org --no-cache-dir -r requirements.txt

# Install NLTK stopwords
RUN python -c "import nltk; nltk.download('stopwords')"

EXPOSE 80

ENTRYPOINT ["python", "build_topic_model_browser.py"]
