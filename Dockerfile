FROM python:slim


WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY * /app

CMD streamlit run streamlitApp_1.py