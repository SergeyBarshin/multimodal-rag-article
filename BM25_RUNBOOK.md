# BM25 baseline: как восстановить эксперимент

## 1) Что сейчас в проекте
- `indices/*.bin` есть (старые dense-индексы сохранены).
- `data/` пустая, поэтому метрики сейчас посчитать нельзя.
- Для BM25 GPU не нужен, только CPU.

## 2) Минимум данных для оценки
Нужны файлы:
- `data/theory.json`
- `data/images.json`
- `data/benchmark_final.json`

## 3) Восстановление данных
1. Восстанови корпус:
```bash
python robust_parser.py
```
2. Сгенерируй `benchmark_final.json` в Colab из `generate_benchmark.ipynb` (Qwen2.5-VL, GPU).

Рекомендуется сразу зафиксировать seed в ноутбуке генерации, чтобы потом можно было повторить результаты:
```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

## 4) Подсчёт метрик BM25
Установка зависимостей:
```bash
pip install -r requirements-bm25.txt
```

Расчёт метрик (по вопросам):
```bash
python bm25_eval.py --data-dir data --query-field question --k 5,10 --output-json bm25_metrics_question.json
```

Опционально, расчёт по ответам (oracle-режим):
```bash
python bm25_eval.py --data-dir data --query-field answer --k 5,10 --output-json bm25_metrics_answer.json
```

## 5) Что считает скрипт
- `Recall@k` (hit-rate, как в твоём ноутбуке)
- `MRR@k`
- `nDCG@k`

Метрики считаются:
- `overall`
- по типам: `text`, `image`, `hybrid`

Скрипт автоматически патчит gold ids:
- если вопрос про изображение, добавляет связанный `preceding_text_id` из `images.json`.

## 6) Как пересчитать остальные метрики (dense + hybrid)
Это твой основной эксперимент из `evaluation.ipynb`.

### Шаг 1. Подготовка в Colab
1. Подключи Drive.
2. Положи в `BASE_DIR` папки:
- `data/` с `theory.json`, `images.json`, `benchmark_final.json`
- `indices/` (или старые, или заново пересобранные)

### Шаг 2. Если нужно пересобрать индексы
Запусти `build_indices.ipynb` полностью. На выходе должны быть:
- `idx_text_rubert.bin`
- `idx_text_user_bge.bin`
- `idx_siglip.bin` + `idx_siglip_ids.json`
- `idx_jina.bin` + `idx_jina_ids.json`
- `idx_clip_openai.bin` + `idx_clip_openai_ids.json` (может быть пустым, если часть картинок не прошла)
- `idx_bad_siglip_full.bin` + `idx_bad_siglip_full_ids.json`
- `idx_jina_full.bin` + `idx_jina_full_ids.json`

### Шаг 3. Пересчёт dense/hybrid метрик
Запусти `evaluation.ipynb` после проверки путей:
- `BASE_DIR`
- `DATA_DIR`
- `INDICES_DIR`

Сценарии в ноутбуке:
- `Vision (Semantic)` (query=`answer`) для `SigLIP/Jina/CLIP`
- `RAG (Question)` (query=`question`) для `BGE`
- `Hybrid` (RRF fusion: `BGE + vision`)

### Шаг 4. Проверка корректности перед финальными цифрами
- В логе должен печататься размер бенчмарка > 0.
- Должны загрузиться все нужные FAISS индексы.
- Для image/hybrid вопросов должен примениться патч `image_id -> preceding_text_id`.
- Запусти минимум 2 прогона и сравни отклонение (особенно если benchmark пересобран заново).

### Шаг 5. Свести итоговую таблицу в статье
Добавь BM25 как отдельный baseline рядом с dense/hybrid:
- `BM25 (question)` из `bm25_metrics_question.json`
- `BM25 (answer)` из `bm25_metrics_answer.json` (как верхняя граница качества ретривера)

Для сравнения с ноутбуком `evaluation.ipynb` используй в первую очередь `Recall@5`.
