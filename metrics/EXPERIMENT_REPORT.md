# 1. Краткий итог

На текущем датасете (`697` вопросов) текстовый и гибридный поиск существенно сильнее чисто визуального zero-shot поиска.  
`SigLIP` в режиме semantic-vision дает низкое качество (`Recall@5 = 3.96%`), `OpenAI CLIP` — `0.00%`.  
`BGE` как текстовый ретривер показывает сильный baseline (`Recall@5 = 62.12%`), а гибрид `BGE + OpenAI CLIP` дает лучший результат в этом прогоне (`Recall@10 = 66.43%`).

BM25 как лексический baseline также конкурентен на mixed-наборе:

- `overall Recall@5 = 57.76%`, `MRR@5 = 0.4869`, `nDCG@5 = 0.5095`;
- по `text/hybrid` срезам BM25 сильный, по `image` срезу — `0.00%` (ожидаемо для чисто текстового матчера без визуального сигнала).

Отдельно: эксперименты с `Jina CLIP` в этом прогоне невалидны из-за runtime-ошибки загрузки модели (см. `dense_hybrid_error_summary.csv`).

---

## 2. Методология и Данные (Methodology)

### А. Сбор данных (Data Mining)

Источник: курс лекций по ТОЭ (`toehelp.ru`).

- Метод: HTML-парсинг с сохранением структуры и связей.
- Объем текущего прогона:
  - `data/theory.json`: **2497** текстовых фрагментов;
  - `data/images.json`: **3065** изображений;
  - `data/benchmark_final.json`: **697** QA-примеров.
- Контекстная связь: для изображений сохраняется `preceding_text_id` (используется для привязки схемы к тексту).

### Б. Архитектура индексов (Indexing Strategies)

Использованы три класса ретриверов:

1. Text:

- `deepvk/USER-bge-m3` + FAISS (`idx_text_user_bge.bin`).

2. Visual:

- `google/siglip-base-patch16-224` (`idx_siglip.bin`);
- `openai/clip-vit-base-patch32` (`idx_clip_openai.bin`);
- `jinaai/jina-clip-v1` (`idx_jina.bin`, в данном прогоне с runtime failure).

3. Lexical baseline:

- `BM25` по текстовому корпусу (`theory.json`).

### В. Протокол оценки (Evaluation Protocol)

Считались метрики:

- `Recall@k`
- `MRR@k`
- `nDCG@k`

Текущий конфиг прогона:

- Vision: `k=5`
- Text-only BGE: `k=5`
- Hybrid: `k=10`
- BM25: `k=5` и `k=10` (overall + type slices)

---

## 3. Результаты Экспериментов (Results)

### 3.1 Dense/Hybrid (из `metrics/dense_hybrid_metrics.csv`)

| Mode              | Model             |  k  | Recall |    MRR |   nDCG | Hits/Total |
| :---------------- | :---------------- | :-: | -----: | -----: | -----: | ---------: |
| Vision (Semantic) | SigLIP            |  5  | 0.0396 | 0.0236 | 0.0169 |   13 / 328 |
| Vision (Semantic) | OpenAI CLIP       |  5  | 0.0000 | 0.0000 | 0.0000 |    0 / 328 |
| RAG (Question)    | Text Only (BGE)   |  5  | 0.6212 | 0.5396 | 0.4829 |  433 / 697 |
| RAG (Question)    | BGE + SigLIP      | 10  | 0.6255 | 0.5216 | 0.4702 |  436 / 697 |
| RAG (Question)    | BGE + OpenAI CLIP | 10  | 0.6643 | 0.5454 | 0.4949 |  463 / 697 |

Примечание: строки по `Jina` отсутствуют в итоговой таблице из-за runtime failure.

### 3.2 BM25 (из `bm25_metrics_question.csv`)

| Group   |  k  | Recall |    MRR |   nDCG | Count |
| :------ | :-: | -----: | -----: | -----: | ----: |
| overall |  5  | 0.5776 | 0.4869 | 0.5095 |   696 |
| overall | 10  | 0.6236 | 0.4932 | 0.5245 |   696 |
| text    |  5  | 0.7182 | 0.5863 | 0.6191 |   369 |
| text    | 10  | 0.7886 | 0.5960 | 0.6422 |   369 |
| hybrid  |  5  | 0.7026 | 0.6285 | 0.6471 |   195 |
| hybrid  | 10  | 0.7333 | 0.6326 | 0.6570 |   195 |
| image   |  5  | 0.0000 | 0.0000 | 0.0000 |   132 |
| image   | 10  | 0.0000 | 0.0000 | 0.0000 |   132 |

---

## 4. Обсуждение и Выводы (Discussion)

### Почему визуальный zero-shot сработал слабо

1. Domain gap: общедоменные vision-энкодеры плохо кодируют топологию электрических схем.
2. Детальность задачи: вопросы требуют точной структуры соединений, что теряется в глобальном image embedding.

### Почему текстовый baseline силён

`BGE` и `BM25` эффективно извлекают релевантный контекст из окружающего текста, где схема уже объясняется словами.  
На этом наборе это дает высокий practical recall без fine-tuning визуальной части.

### Ключевой факт по стабильности эксперимента

`Jina CLIP` сейчас нестабилен в окружении (runtime error на инициализации), поэтому выводы по visual-блоку валидны только для `SigLIP` и `OpenAI CLIP`.

### Рекомендации

1. Привести все сравнения к одинаковому `k` (лучше отдельные таблицы для `k=5` и `k=10`).
2. Зафиксировать версии `transformers/torch` и `revision` для моделей с `trust_remote_code`.
3. Для visual retrieval делать domain adaptation (fine-tuning на схемах ГОСТ).
4. Оставлять текстовый контур retrieval как baseline production-path.

---

## 5. Структура репозитория (Codebase)

```text
├── parser_dataset.ipynb
├── build_indices.ipynb
├── generate_benchmark.ipynb
├── evaluation.ipynb
├── bm25_eval.py
├── metrics_analytics.ipynb
├── data/
│   ├── theory.json
│   ├── images.json
│   └── benchmark_final.json
└── indices/
    ├── idx_text_user_bge.bin
    ├── idx_siglip.bin
    ├── idx_clip_openai.bin
    └── *_ids.json
```
