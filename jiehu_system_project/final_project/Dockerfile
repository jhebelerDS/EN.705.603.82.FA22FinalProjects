FROM python:3.8
COPY .  ./
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade scikit-learn
CMD ["main.py"]
ENTRYPOINT ["python"]