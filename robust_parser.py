import os
import json
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from io import BytesIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tqdm import tqdm 

# --- КОНФИГУРАЦИЯ ---
BASE_DOMAIN = "https://www.toehelp.ru"
URL_TEMPLATE = "https://www.toehelp.ru/theory/toe/lecture{0:02d}/lecture{0:02d}.html"
START_LEC = 1
END_LEC = 43

# Пути (подставьте свои, если отличаются)
DATA_DIR = "data" 
IMG_DIR = os.path.join(DATA_DIR, "images")

os.makedirs(IMG_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# --- НАСТРОЙКА СЕССИИ С ПОВТОРАМИ (Retries) ---
def get_session():
    session = requests.Session()
    retry = Retry(
        total=5,              # Количество попыток
        read=5,               # Попыток при ошибке чтения
        connect=5,            # Попыток при ошибке соединения
        backoff_factor=1,     # Пауза между попытками (1с, 2с, 4с...)
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update(HEADERS)
    return session

# Глобальная сессия
http = get_session()

def get_soup(url):
    try:
        # Увеличили timeout до 30 секунд
        r = http.get(url, timeout=30)
        if r.status_code != 200:
            print(f"⚠️ Status {r.status_code} for {url}")
            return None
        
        # Фикс кодировки для старого сайта
        return BeautifulSoup(r.text, 'html.parser')
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def process_image(img_url, local_path_png):
    try:
        # Тоже используем сессию с ретраями
        r = http.get(img_url, timeout=20)
        if r.status_code != 200: return False

        img_bytes = BytesIO(r.content)
        img = Image.open(img_bytes)
        img.seek(0) # Первый кадр

        # Фон для прозрачности
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg.paste(img.convert('RGBA'), mask=img.convert('RGBA'))
        else:
            bg.paste(img)

        bg.save(local_path_png, "PNG")
        return True
    except Exception as e:
        # print(f"Img Error: {e}") 
        return False

def load_existing_data():
    """Загружает уже скачанные данные, чтобы не дублировать."""
    t_data, i_data = [], []
    t_path = os.path.join(DATA_DIR, "theory.json")
    i_path = os.path.join(DATA_DIR, "images.json")
    
    if os.path.exists(t_path):
        try:
            with open(t_path, 'r', encoding='utf-8') as f:
                t_data = json.load(f)
        except: pass
        
    if os.path.exists(i_path):
        try:
            with open(i_path, 'r', encoding='utf-8') as f:
                i_data = json.load(f)
        except: pass
        
    return t_data, i_data

def save_data(theory_data, images_data):
    """Сохраняет данные на диск"""
    with open(os.path.join(DATA_DIR, "theory.json"), 'w', encoding='utf-8') as f:
        json.dump(theory_data, f, ensure_ascii=False, indent=4)
    with open(os.path.join(DATA_DIR, "images.json"), 'w', encoding='utf-8') as f:
        json.dump(images_data, f, ensure_ascii=False, indent=4)

def parse_lectures():
    # 1. Загружаем то, что уже есть
    theory_data, images_data = load_existing_data()
    
    # Определяем, какие лекции уже готовы
    processed_lectures = set(item['lecture_id'] for item in theory_data)
    print(f"📂 Уже скачано лекций: {len(processed_lectures)}")

    print(f"🚀 Парсинг лекций {START_LEC}-{END_LEC}...")

    for i in range(START_LEC, END_LEC + 1):
        lec_id = f"L{i:02d}"
        
        # --- SKIP LOGIC: Если лекция уже есть в theory.json, пропускаем запрос ---
        if lec_id in processed_lectures:
            # print(f"⏭️ Skipping {lec_id} (already exists)")
            continue

        lec_url = URL_TEMPLATE.format(i)
        print(f"⏳ Processing {lec_id}: {lec_url} ...", end="\r")

        soup = get_soup(lec_url)
        if not soup: 
            print(f"\n❌ Failed to parse {lec_id}")
            continue

        # Временные списки для текущей лекции
        current_theory = []
        current_images = []
        
        last_text_id = None
        paragraphs = soup.find_all('p')

        for p_idx, p in enumerate(paragraphs):
            text = p.get_text(" ", strip=True)
            current_text_id = None

            # Текст
            if len(text) > 1:
                current_text_id = f"{lec_id}_p{p_idx}"
                current_theory.append({
                    "id": current_text_id,
                    "text": text,
                    "lecture_id": lec_id,
                    "source_url": lec_url
                })
                last_text_id = current_text_id

            # Картинки
            imgs = p.find_all('img')
            for img_idx, img in enumerate(imgs):
                src = img.get('src')
                if not src: continue

                # Логика URL
                if src.startswith("./theory"):
                    img_url = BASE_DOMAIN + src.lstrip('.')
                elif src.startswith("http"):
                    img_url = src
                else:
                    img_url = urljoin(lec_url, src)

                local_name = f"{lec_id}_p{p_idx}_img{img_idx}.png"
                local_path = os.path.join(IMG_DIR, local_name)

                # Проверка файла на диске
                if not os.path.exists(local_path):
                    success = process_image(img_url, local_path)
                    if not success: continue

                caption = img.get('alt', '')
                if not caption and len(text) < 300 and "Рис" in text:
                    caption = text

                linked_text_id = current_text_id if current_text_id else last_text_id

                current_images.append({
                    "id": f"{lec_id}_p{p_idx}_img{img_idx}",
                    "path": local_path,
                    "caption": caption,
                    "lecture_id": lec_id,
                    "preceding_text_id": linked_text_id
                })

        # Добавляем данные текущей лекции в общий список
        theory_data.extend(current_theory)
        images_data.extend(current_images)
        
        # --- ВАЖНО: Сохраняем после каждой лекции ---
        save_data(theory_data, images_data)
        
        # Небольшая пауза, чтобы не нагружать сервер (хотя retry спасет)
        time.sleep(0.5)

    print(f"\n✅ Парсинг завершен!")
    print(f"   Всего текста: {len(theory_data)}")
    print(f"   Всего картинок: {len(images_data)}")
    print(f"   Папка: {os.path.abspath(DATA_DIR)}")

if __name__ == "__main__":
    parse_lectures()