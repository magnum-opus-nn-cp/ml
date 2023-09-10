import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, BertConfig
from captum.attr import LayerIntegratedGradients
import re
import torch
import numpy as np
from collections import Counter
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from  matplotlib.colors import LinearSegmentedColormap
from catboost import CatBoostClassifier

from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from  matplotlib.colors import LinearSegmentedColormap

from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer('sentence-transformers/LaBSE')


catboost = CatBoostClassifier().load_model('catboost')

def get_embs(text):
    embeddings = sentence_model.encode(text)
    return embeddings

cmap = LinearSegmentedColormap.from_list('rg',["w", "g"], N=512)
mstm = Mystem()


with open('vectorizer.pickle', 'rb') as file:
    model_tfidf = pickle.load(file)

with open('tree.pickle', 'rb') as file:
    cls = pickle.load(file)


def resolve_text(tokens, text):
    words = text.split()
    tokens_values = list(map(lambda tok: tok[0], tokens))
    tokens_metrics = list(map(lambda tok: tok[1], tokens))
    resolved = []
    for i, word in enumerate(words):
        try:
            if mstm.lemmatize(word)[0] in tokens_values:
                try:
                    value = tokens_metrics[tokens_values.index(mstm.lemmatize(word)[0])]
                    #color = from_abs_to_rgb(min(tokens_metrics), max(tokens_metrics), value)
                    resolved.append(f'<span data-value="{(value - min(tokens_metrics))/ max(tokens_metrics)}">{word}</span>')
                except:
                    resolved.append(word)
            else:
                resolved.append(word)
        except:
            resolved.append(word)
    return ' '.join(resolved)


def process_classify(text):
    if not len(text.replace(' ', '')): return {'ans': 0, 'text': ''}
    try:
        normalized = ''.join(mstm.lemmatize(text)[:-1])
    except: return {'ans': 0, 'text': ''}
    tf_idfed = model_tfidf.transform(np.array([normalized]))[0]

    
    ans = cls.predict(tf_idfed)[0]
    return {'ans': ans, 'text': ""}


def process_embedding(text):
    if not len(text.replace(' ', '')): return {'ans': 0, 'text': ''}
    try:
        normalized = ''.join(mstm.lemmatize(text)[:-1])
    except: return {'ans': 0, 'text': ''}
    tf_idfed = model_tfidf.transform(np.array([normalized]))[0]
    values = []
    for i in range(5000):
        values.append(tf_idfed.todense()[0, i])
    
    important_tokens = []
    for i, val in enumerate(values):
        if val > (np.min(values) + np.max(values)) / 3:
            important_tokens.append((val, i))
    tokens = model_tfidf.get_feature_names_out()
    tokens = list(map(lambda x: (tokens[x[1]], x[0]), reversed(sorted(important_tokens))))
    
    ans = cls.predict(tf_idfed)[0]
    text = resolve_text(tokens, text)
    return {'ans': ans, 'text': text}

cmap = LinearSegmentedColormap.from_list('rg',["w", "g"], N=512) 

label2id = {
 'AAA(RU)': 0,
 'AA(RU)': 1, 
 'A+(RU)': 2,
 'A(RU)': 3,
 'A-(RU)': 4,
 'BBB+(RU)': 5,
 'BBB(RU)': 6, 
 'AA+(RU)': 7,
 'BBB-(RU)': 8,
 'AA-(RU)': 9,
 'BB+(RU)': 10, 
 'BB-(RU)': 11, 
 'B+(RU)': 12,
 'BB(RU)': 13, 
 'B(RU)': 14,
 'B-(RU)': 15, 
 'C(RU)': 16
}
id2label = {0: 'AAA(RU)',
 1: 'AA(RU)',
 2: 'A+(RU)',
 3: 'A(RU)',
 4: 'A-(RU)',
 5: 'BBB+(RU)',
 6: 'BBB(RU)',
 7: 'AA+(RU)',
 8: 'BBB-(RU)',
 9: 'AA-(RU)',
 10: 'BB+(RU)',
 11: 'BB-(RU)',
 12: 'B+(RU)',
 13: 'BB(RU)',
 14: 'B(RU)',
 15: 'B-(RU)',
 16: 'C(RU)'}

cmap = LinearSegmentedColormap.from_list('rg',["w", "g"], N=512) 


from math import inf
from annoy import AnnoyIndex
import numpy as np
import pickle



def get_distance(emb1, emb2):
    emb2 /= np.sum(emb2**2)
    emb1 /= np.sum(emb1**2)
    return 1 / abs(np.dot(emb2-emb1, emb1-emb2))


with open('new_embeddings.pickle', 'rb') as file:
    new_embeddings = pickle.load(file)

with open('annoy_labels.pickle', 'rb') as file:
    labels = pickle.load(file)

with open('n_labels.pickle', 'rb') as file:
    n_labels = pickle.load(file)

index = AnnoyIndex(768, 'angular')
index.load('nearest.annoy')


def get_nearest_value(embeddings):
    items = list(map(lambda x: (
            labels[x], 
            get_distance(embeddings, new_embeddings[x]),
            list(n_labels)[x]
        ), 
        index.get_nns_by_vector(embeddings, 20)
    ))
    weights = np.array([0 for _ in range(17)])
    refs = [[] for _ in range(17)]
    s = 0
    for item in items:
        if item[1] == inf:
            return id2label[item[0]], 100, [item[2]]
        s += item[1]
        weights[item[0]] += item[1]
        refs[item[0]].append(item[2])
    return id2label[np.argmax(weights)], (weights[np.argmax(weights)] / s) * 100, refs[np.argmax(weights)]


def to_rgb(vals):
    return f'rgb({int(vals[0]*255)}, {int(vals[1]*255)}, {int(vals[2]*255)})'

def from_abs_to_rgb(min, max, value):
    return to_rgb(cmap((value - min)/ max))


def get_nns_tokens(encoding, attrs, predicted_id):
    current_array = map(
        lambda x: (tokenizer.convert_ids_to_tokens(encoding['input_ids'][0][x[0]-5:x[0]+5]), x[1]), 
        list(
            reversed(
                sorted(
                    enumerate(
                        attrs[0][predicted_id].numpy()
                    ), 
                    key=lambda x: x[1]
                )
            )
        )[0:10]
    )
    return list(current_array)

def get_description_interpreting(attrs, predicted_id):
    attrs = attrs.detach().numpy()
    positive_weights = attrs[0][predicted_id]
    negative_weights = [0 for _ in range(len(positive_weights))]
    for i in range(len(attrs[0])):
        if i == predicted_id: continue
        for j in range(len(attrs[0][i])):
            negative_weights[j] += attrs[0][i][j]
    for i in range(len(negative_weights)):
        negative_weights[i] /= len(attrs[0]) - 1
    
    return {
        'positive_weights': (
            positive_weights, 
            {
                'min': np.min(positive_weights),
                'max': np.max(positive_weights)
            }
        ),
        'negative_weights': (
            negative_weights,
            {
                'min': min(negative_weights),
                'max': max(negative_weights)
            }
        )
    }

def transform_token_ids(func_data, token_ids, word):
    tokens = list(map(lambda x: tokenizer.convert_ids_to_tokens([x])[0].replace('##', ''), token({'text': clean(word)})['input_ids'][0]))
    weights = [func_data['positive_weights'][0][i] for i in token_ids]
    wts = []
    for i in range(len(weights)):
        if weights[i] > 0:
            #color = from_abs_to_rgb(func_data['positive_weights'][1]['min'], func_data['positive_weights'][1]['max'], weights[i])
            mn = max(func_data['positive_weights'][1]['min'], 0)
            mx = func_data['positive_weights'][1]['max']
            wts.append((weights[i] - mn)/ mx)
            #word = word.lower().replace(tokens[i], f'<span data-value="{(weights[i] - mn)/ mx}">{tokens[i]}</span>')
    try:
        if sum(wts) / len(wts) >= 0.2:
            return f'<span data-value={sum(wts) / len(wts)}>{word}</span>'
    except: pass
    return word


def build_text(tokens, func_data, current_text):
    splitted_text = current_text.split()
    splitted_text_iterator = 0
    current_word = ''
    current_word_ids = []
    for i, token in enumerate(tokens):
        decoded = tokenizer.convert_ids_to_tokens([token])[0]
        if decoded == '[CLS]': continue
        if not len(current_word):
            current_word = decoded
            current_word_ids.append(i)
        elif decoded.startswith('##'):
            current_word += decoded[2:]
            current_word_ids.append(i)
        else:
            while clean(splitted_text[splitted_text_iterator]) != current_word:
                splitted_text_iterator += 1
            current_word = decoded
            splitted_text[splitted_text_iterator] = transform_token_ids(func_data, current_word_ids, splitted_text[splitted_text_iterator])
            current_word_ids = []
    return ' '.join(splitted_text)

def squad_pos_forward_func(inputs, token_type_ids=None, attention_mask=None, position=0):
    pred = predict(inputs.to(torch.long), token_type_ids.to(torch.long), attention_mask.to(torch.long))
    pred = pred[position]
    return pred.max(1).values

def predict_press_release(input_ids, token_type_ids, attention_mask):
    encoding = {
        'input_ids': input_ids.to(model.device),
        'token_type_ids': token_type_ids.to(model.device),
        'attention_mask': attention_mask.to(model.device)
    }
    outputs = model(**encoding)
    return outputs


def clean(text):
    text = re.sub('[^а-яё ]', ' ', str(text).lower())
    text = re.sub(r" +", " ", text).strip()
    return text


def get_description_interpreting(attrs):
    positive_weights = attrs
    return {
        'positive_weights': (
            positive_weights, 
            {
                'min': np.min(positive_weights),
                'max': np.max(positive_weights)
            }
        ),
    }


def predict(input_ids, token_type_ids, attention_mask):
    encoding = {
        'input_ids': input_ids.to(model.device),
        'token_type_ids': token_type_ids.to(model.device),
        'attention_mask': attention_mask.to(model.device)
    }
    outputs = model(**encoding)
    return outputs


def batch_tokenize(text):
    splitted_text = text.split()
    current_batch = splitted_text[0]
    batches = []
    for word in splitted_text[1:]:
        if len(tokenizer(current_batch + ' ' + word)['input_ids']) < 512:
            current_batch += ' ' + word
        else:
            batches.append({
                'text': current_batch
            })
            current_batch = word
    return batches + [{'text': current_batch}]


def token(text):
    return tokenizer(text['text'], padding=True, truncation=True, max_length=512, return_tensors='pt')


def tfidf_classify(data):
    if not len(data.data): return ''
    data = list(map(lambda x: x['text'], batch_tokenize(clean(data.data))))
    predicted_labels = []
    predicted_text = ""
    for item in data:
        predicted_labels.append(process_classify(item)['ans'])
    ans = Counter(predicted_labels).most_common()[0][0]
    score = len(list(filter(lambda x: x == ans, predicted_labels))) / len(predicted_labels)
    ans = id2label[ans]
    return {'answer': ans, 'text': predicted_text, 'metric': score, 'extendingLabels': list(map(lambda x: id2label[x], predicted_labels))}


def tfidf_embeddings(data):
    if not len(data.data): return ''
    data = list(map(lambda x: x['text'], batch_tokenize(clean(data.data))))
    predicted_labels = []
    predicted_text = ""
    for item in data:
        ans = process_embedding(item)
        predicted_labels.append(ans['ans'])
        predicted_text += ans['text'] + ' '
    ans = Counter(predicted_labels).most_common()[0][0]
    print(ans, predicted_text)
    return {'answer': id2label[ans], 'text': predicted_text}


def bert_classify(data):
    data = clean(data)
    predicted = []
    text = ''
    batched = batch_tokenize(data)

    for b in batched:
        print(len(predicted))
        embs = token(b)
        answer = predict_press_release(
                    embs['input_ids'], embs['token_type_ids'], embs['attention_mask']
                    ).logits[0]
        answer = torch.softmax(answer, dim=-1).detach().numpy()
        answer_score = np.max(answer)
        predicted.append(
            [id2label[np.argmax(answer)],
            float(answer_score)]
            )
    ans = {'AA(RU)': [0]}
    for i in predicted:
        if i[0] not in ans.keys():
            ans.update({i[0]: [i[1]]})
        else:
            ans[i[0]].append(i[1])
    selected = 'AA(RU)'
    score = 0
    for candidate in ans.keys():
        if sum(ans[candidate]) / len(ans[candidate]) > score:
            score = sum(ans[candidate]) / len(ans[candidate])
            selected = candidate
        elif sum(ans[candidate]) / len(ans[candidate]) == score and len(ans[candidate]) > len(ans):
            selected = candidate
    return {
        'answer': selected, 
        'text': text,
        'longAnswer': predicted,
        'metric': score
    }


def bert_embeddings(data):
    data = clean(data)
    predicted = []
    text = ''
    batched = batch_tokenize(data)
    for b in batched:
        embs = token(b)
        predicted.append(np.argmax(predict_press_release(embs['input_ids'], embs['token_type_ids'], embs['attention_mask']).logits.detach().numpy()[0]))
        attrs = lig.attribute(embs['input_ids'], additional_forward_args=(embs['attention_mask'], embs['token_type_ids'], 0))
        attrs = np.array(list(map(lambda x: x.sum(), attrs[0])))
        descr = get_description_interpreting(attrs)
        text += build_text(embs['input_ids'][0], descr, b['text']) + ' '
    return {'answer': id2label[Counter(predicted).most_common()[0][0]], 'text': text}


config = BertConfig.from_json_file("./akra_model/checkpoint/config.json")

model = AutoModelForSequenceClassification.from_pretrained(
    "./akra_model/checkpoint", config=config
)
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")

lig = LayerIntegratedGradients(squad_pos_forward_func, model.bert.embeddings)

app = FastAPI()

class Predict(BaseModel):
    data: str

class ListPredict(BaseModel):
    data: list


@app.post('/predict')
def predict_(data: Predict):
    return bert_classify(data)

@app.post('/bert/process')
def predict_f(data: Predict):
    return bert_classify(data)
  

@app.get('/interpret')
def interpret():
    pass


@app.post('/tfidf/process')
def tfidf_res(data: Predict):
    return tfidf_classify(data)

@app.post('/tfidf/batch')
def tfidf_batch(data: ListPredict):
    res = []
    for item in data.data:
        res.append(tfidf_classify(Predict(data=item)))
    return res

@app.post('/bert/batch')
def bert_batch(data: ListPredict):
    res = []
    for item in data:
        res.append(bert_classify({'data': item}))
    return res

@app.post('/bert/describe')
def bert_describe(data: Predict):
    return bert_embeddings(data)

@app.post('/tfidf/describe')
def tfidf_describe(data: Predict):
    return tfidf_embeddings(data)


def get_nearest_service(data: Predict):
    data = clean(data.data)
    batched = batch_tokenize(data)
    res = []
    scores = {}
    for key in id2label.values():
        scores.update({key: []})
    for batch in batched:
        features = list(get_nearest_value(get_embs(batch['text'])))
        features[0] = features[0]
        features[1] /= 100
        scores[features[0]].append(features[1] if features[1] < 95 else 100)
        res.append(
            {
                'text': batch['text'],
                'features': features
            }
        )
    mx = 0
    label = 'AA(RU)'
    for key in scores.keys():
        try:
            if (sum(scores[key]) / len(scores[key])) > mx:
                label = key
                mx = (sum(scores[key]) / len(scores[key]))
            if (sum(scores[key]) / len(scores[key])) == mx:
                if len(scores[key]) > len(scores[label]):
                    label = key
        except: pass
    return {'detailed': res, 'metric': mx, 'answer': label}


@app.post('/nearest/nearest')
def proccess_text(data: Predict):
    return get_nearest_service(data)


@app.post('/catboost')
def catboost_process(data: Predict):
    tfidf = tfidf_classify(data)
    bert = bert_classify(data)
    nearest = get_nearest_service(data)

    inputs = [label2id[tfidf['answer']], tfidf['metric'], bert['metric'], label2id[bert['answer']], nearest['metric'], label2id[nearest['answer']]]
    catboost_answer = id2label[catboost.predict([inputs])[0][0]]
    return {
        'bert': bert,
        'tfidf': tfidf,
        'nearest': nearest,
        'total': catboost_answer
    }