.PHONY: install run train

install:
	pip install -r requirements.txt

run:
	streamlit run src/app.py

train:
	python src/train.py