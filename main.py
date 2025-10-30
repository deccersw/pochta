# БЛОК 1: Импорт библиотек
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import umap.umap_ as umap
import hdbscan
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import warnings
import spacy
import os

warnings.filterwarnings('ignore')

# БЛОК 2: Загрузка и предобработка данных
print("=== БЛОК 2: Загрузка данных ===")

# Укажите путь к вашему файлу
file_path = "/Users/danial2006/Хакатон/Пишу тебе. Корпус для хакатона (2024).xlsx"  # Измените на ваш путь к файлу

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Файл {file_path} не найден. Укажите правильный путь.")

df = pd.read_excel(file_path, sheet_name=0)

print(f"Исходный размер датасета: {len(df)} записей")

# Фильтруем только русские тексты
df = df[df["Язык текста открытки"] == "русский"].dropna(subset=["Текст открытки"]).reset_index(drop=True)
print(f" После фильтрации русских текстов: {len(df)} записей")

# Обрабатываем даты более аккуратно
print("\n🔧 Обрабатываем даты...")

# Создаем копию столбца с датами для безопасной обработки
df['date_processed'] = pd.to_datetime(df['Дата открытки (нормализованная)'], errors='coerce')

# Считаем статистику по датам
valid_dates = df['date_processed'].notna()
print(f"Корректных дат: {valid_dates.sum()}")
print(f"Некорректных/пропущенных дат: {len(df) - valid_dates.sum()}")

if valid_dates.any():
    min_date = df.loc[valid_dates, 'date_processed'].min()
    max_date = df.loc[valid_dates, 'date_processed'].max()
    print(f"Диапазон дат: {min_date} - {max_date}")
else:
    print("Нет корректных дат для анализа")

# Добавляем год и десятилетие только для корректных дат
df['year'] = df['date_processed'].dt.year
df['decade'] = (df['year'] // 10) * 10

# Показываем распределение по десятилетиям
decade_counts = df[df['decade'].notna()]['decade'].value_counts().sort_index()
print("\nРаспределение по десятилетиям:")
for decade, count in decade_counts.items():
    print(f"   {int(decade)}-е: {count} записей")

print(f"\nГотово к анализу: {len(df)} русских открыток")

print(df[['Текст открытки', 'date_processed', 'year', 'decade']].head(3))

# БЛОК 3: ПРЕДОБРАБОТКА ТЕКСТОВ
print("\n=== БЛОК 3: Предобработка текстов ===")

def normalize_text(text):
    """Мягкая нормализация текста"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Замена дореволюционных букв
    text = re.sub("ѣ", "е", text)
    text = re.sub("і", "и", text)
    text = re.sub("ѳ", "ф", text)
    text = re.sub("ъ(?=\s|$)", "", text)
    # Удаление лишних символов
    text = re.sub("[^а-яё\s]", " ", text)
    text = re.sub("\s+", " ", text).strip()
    return text

# Применяем нормализацию
df["text_clean"] = df["Текст открытки"].apply(normalize_text)

print("Примеры очищенных текстов:")
for i in range(2):
    original = df["Текст открытки"].iloc[i][:100] + "..."
    cleaned = df["text_clean"].iloc[i][:100] + "..."
    print(f"\nПример {i+1}:")
    print(f"Оригинал: {original}")
    print(f"Очищенный: {cleaned}")

print(f"\nОбработано {len(df)} текстов")

# БЛОК 4: ИЗВЛЕЧЕНИЕ N-ГРАММ
print("\n=== БЛОК 4: Извлечение n-грамм ===")

# Используем CountVectorizer для извлечения биграмм и триграмм
vectorizer = CountVectorizer(
    analyzer="word",
    ngram_range=(2, 4),  # Биграммы, триграммы и четырехграммы
    min_df=2,  # Фраза должна встречаться минимум в 2 документах
    max_df=0.8,  # Исключаем слишком частые фразы
    token_pattern=r"(?u)\b[а-яё]{2,}\b"
)

# Обучаем векторйзер и преобразуем тексты
X = vectorizer.fit_transform(df["text_clean"])
phrases = vectorizer.get_feature_names_out()
frequencies = X.sum(axis=0).A1

print(f"Извлечено {len(phrases)} n-грамм")

# Создаем DataFrame с фразами и частотами
phrases_df = pd.DataFrame({
    "phrase": phrases,
    "frequency": frequencies
}).sort_values("frequency", ascending=False)

print("\n🏆 Топ-20 самых частых фраз:")
print(phrases_df.head(20))

# Сохраняем матрицу для дальнейшего использования
phrase_doc_matrix = X

# БЛОК 5: POS-ФИЛЬТРАЦИЯ N-ГРАММ С ИСПОЛЬЗОВАНИЕМ SPACY
print("\n=== БЛОК 5: POS-фильтрация n-грамм с использованием spaCy ===")

# Проверяем наличие модели spaCy
try:
    nlp = spacy.load("ru_core_news_md")
    print("Модель spaCy загружена!")
except OSError:
    print("Модель 'ru_core_news_md' не найдена.")
    print("Установите модель выполнив в терминале:")
    print("python -m spacy download ru_core_news_md")
    exit(1)

# Функция для получения POS-паттерна
def get_pos_pattern(phrase):
    """Возвращает строку с POS-тегами для фразы"""
    doc = nlp(phrase)
    return " ".join([t.pos_ for t in doc if not t.is_punct])

print("Анализируем части речи для n-грамм...")

# Получаем POS-паттерны для всех фраз
pos_patterns = []
for phrase in tqdm(phrases):
    pos_patterns.append(get_pos_pattern(phrase))

# Создаем DataFrame с фразами, частотами и POS-паттернами
df_phr = pd.DataFrame({
    "phrase": phrases, 
    "freq": frequencies, 
    "pattern": pos_patterns
})

print(f"Проанализировано {len(df_phr)} фраз")

# Определяем шаблоны для значимых фраз поздравлений
keep_patterns = [
    "VERB NOUN",           # "поздравляю с праздником"
    "VERB ADP NOUN",       # "желаю в новый год" 
    "ADJ NOUN",            # "дорогой друг", "счастливого рождества"
    "PRON ADJ NOUN",       # "мой дорогой друг"
    "NOUN NOUN",           # "день рождения"
    "ADV VERB",            # "крепко целую"
    "VERB ADV",            # "желаю сильно"
    "ADJ ADJ NOUN",        # "дорогой милый друг"
    "VERB PRON NOUN",      # "желаю тебе счастья"
    "ADP NOUN NOUN",       # "с днем рождения"
    "VERB DET NOUN",       # "желаю всего хорошего"
    "ADJ VERB",            # "рад поздравить"
    "NOUN VERB",           # "поздравления принимаю"
]

# Фильтруем фразы по шаблонам
df_phr_filtered = df_phr[df_phr["pattern"].isin(keep_patterns)].copy()

print(f"До фильтрации: {len(df_phr)} фраз")
print(f"После фильтрации: {len(df_phr_filtered)} фраз")

# Анализируем распределение по паттернам
pattern_stats = df_phr_filtered['pattern'].value_counts()
print("\nРаспределение по POS-паттернам:")
for pattern, count in pattern_stats.items():
    print(f"   {pattern}: {count} фраз")

# Обновляем наши переменные для дальнейшего анализа
meaningful_phrases = df_phr_filtered['phrase'].tolist()
meaningful_frequencies = df_phr_filtered['freq'].tolist()
meaningful_phrases_df = df_phr_filtered.sort_values('freq', ascending=False)

print("\nТоп-20 самых частых фраз после POS-фильтрации:")
print(meaningful_phrases_df.head(20))

# Показываем примеры для каждого паттерна
print("\nПримеры фраз по каждому POS-паттерну:")
for pattern in keep_patterns:
    examples = df_phr_filtered[df_phr_filtered['pattern'] == pattern].head(2)
    if len(examples) > 0:
        print(f"\n   {pattern}:")
        for _, row in examples.iterrows():
            print(f"      '{row['phrase']}' (частота: {row['freq']})")

# Визуализация распределения паттернов
print("\nСоздаем визуализации...")

fig_patterns = px.bar(
    pattern_stats.reset_index(),
    x='pattern',
    y='count',
    title='Распределение фраз по POS-паттернам',
    labels={'pattern': 'POS-паттерн', 'count': 'Количество фраз'},
    color='count',
    color_continuous_scale='Viridis'
)

fig_patterns.update_layout(
    xaxis_tickangle=-45,
    height=500
)
fig_patterns.show()

# Создаем treemap для визуализации структуры фраз
if len(df_phr_filtered) > 0:
    fig_treemap = px.treemap(
        df_phr_filtered.nlargest(50, 'freq'),
        path=['pattern', 'phrase'],
        values='freq',
        title='Структура значимых фраз по POS-паттернам',
        color='freq',
        color_continuous_scale='Blues'
    )
    fig_treemap.update_layout(height=600)
    fig_treemap.show()

print(f"\nГотово для дальнейшего анализа: {len(meaningful_phrases)} значимых фраз")