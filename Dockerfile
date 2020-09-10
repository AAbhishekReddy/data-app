FROM python:3.8
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . /app
CMD streamlit run data_app.py --server.port 8501
RUN mkdir -p /root/.streamlit

# RUN apt-get install cuda-cudart

# RUN apt-get install 'ffmpeg'\
#     'libsm6'\ 
#     'libxext6'  -y

RUN echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" 
RUN > ~/.streamlit/config.toml