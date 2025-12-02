import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import sys
import os
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Завантаження змінних середовища
load_dotenv()

# --- НАЛАШТУВАННЯ ---
API_KEY = os.getenv('YOUTUBE_API_KEY')
MAX_RESULTS = 50  # Кількість коментарів для аналізу

# Ініціалізація інструментів
analyzer = SentimentIntensityAnalyzer()
translator = GoogleTranslator(source='auto', target='en')


# --- ФУНКЦІЇ ---

def extract_video_id(url):
    """
    Витягує YouTube ID з різних форматів посилань.
    Підтримує:
    - https://www.youtube.com/watch?v=ID
    - https://youtu.be/ID
    - https://www.youtube.com/shorts/ID
    - Просто ID (якщо користувач ввів тільки його)
    """
    # Якщо це схоже на чистий ID (11 символів, без пробілів), повертаємо як є
    if len(url) == 11 and ' ' not in url and '/' not in url:
        return url

    query = urlparse(url)

    # Випадок 1: youtu.be/ID
    if query.hostname == 'youtu.be':
        return query.path[1:]

    # Випадок 2: youtube.com/watch?v=ID або youtube.com/shorts/ID
    if query.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if query.path == '/watch':
            p = parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
        if query.path[:8] == '/shorts/':  # Підтримка Shorts
            return query.path.split('/')[2]

    # Якщо нічого не спрацювало, повертаємо None
    return None


def clean_text(text):
    """Прибирає зайві пробіли між літерами (T O P -> TOP)"""
    return re.sub(r'(?<=\b\w)\s+(?=\w\b)', '', text)


def analyze_comment(text):
    """Переклад + Аналіз емоцій"""
    try:
        translated = translator.translate(text)
    except:
        translated = text

    if not translated:
        translated = text

    final_text = clean_text(translated)
    scores = analyzer.polarity_scores(final_text)
    compound = scores['compound']

    if compound >= 0.05:
        category = 'Positive'
    elif compound <= -0.05:
        category = 'Negative'
    else:
        category = 'Neutral'

    return compound, category, final_text


def get_data(video_id, api_key, max_results):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
    except Exception as e:
        print(f"\nПомилка підключення до API: {e}")
        return None

    data = []
    print(f"📥 Завантажую коментарі для відео ID: {video_id}...")

    try:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText"
        )
        response = request.execute()

        total = len(response['items'])
        print(f"🔄 Початок обробки {total} коментарів...")

        for i, item in enumerate(response['items']):
            snippet = item['snippet']['topLevelComment']['snippet']
            original_text = snippet['textDisplay']
            author = snippet['authorDisplayName']

            score, category, translated_text = analyze_comment(original_text)

            data.append({
                'Author': author,
                'Original': original_text,
                'Translated': translated_text,
                'Score': score,
                'Category': category
            })

            # Виводимо прогрес у консоль
            print(f"🔄 Оброблено: {i + 1}/{total}", end='\r')

        print("\n✅ Обробку завершено!")
        return pd.DataFrame(data)

    except Exception as e:
        print(f"\nПомилка при отриманні даних (можливо, некоректний ID або закриті коментарі): {e}")
        return None


def show_report(df, video_id):
    avg_score = df['Score'].mean()
    total = len(df)
    pos_count = len(df[df['Category'] == 'Positive'])
    neg_count = len(df[df['Category'] == 'Negative'])
    neu_count = len(df[df['Category'] == 'Neutral'])

    if avg_score > 0.1:
        verdict = "👍 Позитивний"
    elif avg_score < -0.1:
        verdict = "👎 Негативний"
    else:
        verdict = "😐 Нейтральний/Змішаний"

    print("\n" + "=" * 60)
    print(f"📊 ЗАГАЛЬНИЙ ЗВІТ")
    print("=" * 60)
    print(f"🔹 Всього коментарів: {total}")
    print(f"🔹 Середній рейтинг:  {avg_score:.4f} (від -1 до 1)")
    print(f"🔹 Вердикт аудиторії: {verdict}")
    print("-" * 30)
    print(f"💚 Позитивних: {pos_count} ({pos_count / total * 100:.1f}%)")
    print(f"❤️ Негативних: {neg_count} ({neg_count / total * 100:.1f}%)")
    print(f"⚪ Нейтральних: {neu_count} ({neu_count / total * 100:.1f}%)")
    print("=" * 60)

    df_sorted = df.sort_values(by='Score', ascending=False)

    print("\n🏆 ТОП-5 НАЙДОБРІШИХ КОМЕНТАРІВ:")
    for i, row in df_sorted.head(5).iterrows():
        clean_comment = row['Original'][:80].replace('\n', ' ')
        print(f"  [{row['Score']:.2f}] {row['Author']}: {clean_comment}...")

    print("\n🤬 ТОП-5 НАЙЗЛІШИХ КОМЕНТАРІВ:")
    for i, row in df_sorted.tail(5).iterrows():
        clean_comment = row['Original'][:80].replace('\n', ' ')
        print(f"  [{row['Score']:.2f}] {row['Author']}: {clean_comment}...")

    print("\n" + "=" * 60)

    # Графіки
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Аналіз настрою аудиторії (ID: {video_id})', fontsize=16)

    counts = df['Category'].value_counts()
    colors = {'Positive': '#66bb6a', 'Neutral': '#fff176', 'Negative': '#ef5350'}
    pie_colors = [colors.get(k, '#bdbdbd') for k in counts.index]

    if len(counts) > 0:
        axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
                    colors=pie_colors, explode=[0.05] * len(counts))
    axes[0].set_title('Частки емоцій')

    sns.histplot(df['Score'], bins=20, kde=True, ax=axes[1], color='#5c6bc0')
    axes[1].set_title('Розподіл оцінок')
    axes[1].set_xlabel('Негатив (-1) <----> Позитив (+1)')
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()


# --- ГОЛОВНИЙ БЛОК ---
if __name__ == "__main__":
    if not API_KEY:
        print(" Помилка: Не знайдено API ключ! Перевірте файл .env")
        sys.exit()

    # Запит посилання у користувача
    url_input = input("🔗 Вставте посилання на відео YouTube (або просто ID): ").strip()

    # Витягування ID
    video_id = extract_video_id(url_input)

    if video_id:
        df = get_data(video_id, API_KEY, MAX_RESULTS)

        if df is not None:
            show_report(df, video_id)
            # Збереження
            filename = f"report_{video_id}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n💾 Дані збережено у файл '{filename}'")
    else:
        print("Не вдалося розпізнати коректне посилання на YouTube.")