# Экспериментальный протокол: Benchmark -> BM25 -> Все индексы

## 0) Цель
Получить воспроизводимый пайплайн с обязательным сохранением метрик в CSV:
1. создание бенчмарка;
2. оценка BM25;
3. оценка dense/hybrid по всем индексам;
4. расширенный анализ ошибок.

## 1) Подготовка окружения (сервер)
Рабочая директория:
```bash
cd /Users/sergey/Documents/Computer\ Science/Electro_rag_article
```

Создать окружение:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Установить зависимости BM25:
```bash
python -m pip install -r requirements-bm25.txt
```

Для dense/hybrid этапа (FAISS + модели):
```bash
python -m pip install faiss-cpu sentence-transformers transformers accelerate qwen-vl-utils torch pandas tqdm
```

## 2) Создание benchmark (`benchmark_final.json`)
Требуемые входные данные:
- `data/theory.json`
- `data/images.json`
- `data/images/*.png`

Если корпуса нет, восстановить:
```bash
python robust_parser.py
```

Далее открыть `generate_benchmark.ipynb` и выполнить по порядку.

В ноутбуке уже серверные настройки:
- `BASE_DIR` берется из `EE_RAG_BASE_DIR` или `os.getcwd()`;
- `DEVICE` выбирается автоматически;
- параметры генерации можно задавать через env:
  - `BENCHMARK_SEED`
  - `BENCHMARK_TARGET_COUNT`
  - `BENCHMARK_BATCH_SAVE`
  - `BENCHMARK_OUTPUT_FILE`

Пример запуска Jupyter:
```bash
source .venv/bin/activate
jupyter lab
```

Ожидаемый артефакт:
- `data/benchmark_final.json`

## 3) Оценка BM25 + сохранение всех CSV
Запуск:
```bash
source .venv/bin/activate
python bm25_eval.py \
  --data-dir data \
  --query-field question \
  --k 5,10 \
  --output-json bm25_metrics_question.json \
  --output-csv bm25_metrics_question.csv \
  --per-query-csv bm25_per_query_question.csv \
  --errors-csv bm25_errors_question.csv \
  --error-summary-csv bm25_error_summary_question.csv
```

Опционально oracle-режим:
```bash
python bm25_eval.py \
  --data-dir data \
  --query-field answer \
  --k 5,10 \
  --output-json bm25_metrics_answer.json \
  --output-csv bm25_metrics_answer.csv \
  --per-query-csv bm25_per_query_answer.csv \
  --errors-csv bm25_errors_answer.csv \
  --error-summary-csv bm25_error_summary_answer.csv
```

CSV-артефакты BM25:
- `bm25_metrics_*.csv` -> агрегированные метрики (`Recall`, `MRR`, `nDCG`);
- `bm25_per_query_*.csv` -> построчные метрики по каждому query;
- `bm25_errors_*.csv` -> только промахи;
- `bm25_error_summary_*.csv` -> группировка причин промахов.

## 4) Оценка на всех остальных индексах (dense/hybrid)
Нужные файлы в `indices/`:
- `idx_text_rubert.bin`
- `idx_text_user_bge.bin`
- `idx_siglip.bin`, `idx_siglip_ids.json`
- `idx_jina.bin`, `idx_jina_ids.json`
- `idx_clip_openai.bin`, `idx_clip_openai_ids.json`
- `idx_bad_siglip_full.bin`, `idx_bad_siglip_full_ids.json`
- `idx_jina_full.bin`, `idx_jina_full_ids.json`

Если индексов нет или они не согласованы с текущим корпусом, пересобрать через `build_indices.ipynb`.

Оценка выполняется через `evaluation.ipynb`.
После формирования `results` и `df` в ноутбуке обязательно добавить сохранение CSV:
```python
import os
import pandas as pd

OUT_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(OUT_DIR, exist_ok=True)

df.to_csv(os.path.join(OUT_DIR, "dense_hybrid_metrics.csv"), index=False, encoding="utf-8")
```

## 5) Расширенный анализ ошибок (обязательно)
### 5.1 BM25
Использовать:
- `bm25_errors_question.csv`
- `bm25_error_summary_question.csv`

Что проверять:
- доля `no_lexical_match` (вопросы без лексического пересечения);
- доля `short_query` (слишком короткие запросы);
- доля `multi_gold_miss` (сложные multi-gold случаи);
- типы запросов (`text/image/hybrid`) с наибольшим числом промахов.

### 5.2 Dense/Hybrid
В `evaluation.ipynb` добавить лог по каждому запросу в CSV:
```python
error_rows = []
# внутри цикла оценки для каждого query:
# error_rows.append({
#   "model": model_name,
#   "mode": mode_name,
#   "query": query,
#   "type": item["type"],
#   "gold_ids": "|".join(item["gold_ids"]),
#   "pred_ids": "|".join(pred_ids_topk),
#   "hit": int(found)
# })

pd.DataFrame(error_rows).to_csv(
    os.path.join(OUT_DIR, "dense_hybrid_per_query.csv"),
    index=False,
    encoding="utf-8"
)
```

Минимальные срезы анализа:
- ошибки по `type` (`text/image/hybrid`);
- ошибки по модели (`BGE`, `SigLIP`, `Jina`, `CLIP`, гибриды);
- топ-20 самых частых промахов по формулировкам вопросов.

## 6) Финальные файлы для статьи
Минимальный комплект:
- `metrics/dense_hybrid_metrics.csv`
- `bm25_metrics_question.csv`
- `bm25_metrics_answer.csv` (если считали oracle)
- `bm25_error_summary_question.csv`
- `metrics/dense_hybrid_per_query.csv`

Этого достаточно для таблицы метрик и отдельного подраздела "Анализ ошибок".
