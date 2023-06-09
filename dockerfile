FROM python:3.10.11

WORKDIR /AutoGPT

ADD . .


RUN pip install -r requirements.txt


ENV OPENAI_API_KEY=


EXPOSE 5200

CMD ["streamlit run", "zhanxing2.py"]