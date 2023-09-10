### Установить зависимости
`pip3 install -r req.txt`
### Локации документов для обучения
- `catboost-train.ipynb` - файл для обучения кэтбуста
- `bert_train.ipynb` - файл для обучения берта
- `nearest-search-train.ipynb` - файл для обучения поиска близжайших соседей
- `tfidf-train.ipynb` - файл для обучения tf-idf + random forest
### Инференс модели
`uvicorn inference:app --reload --workers 1`
### У мля слишком много больших файлов поэтому мы выложили код с весами моделей на гугл диск https://drive.google.com/drive/folders/1hnWKpZjtQLBbzAE9YsUW_4x-IEb3mFvg?usp=sharing
