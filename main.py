import os
import sys
import re
import io
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from telebot import TeleBot, types
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

# --- НАЛАШТУВАННЯ ---
load_dotenv()
YT_API_KEY = os.getenv('YOUTUBE_API_KEY')
TG_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

if not YT_API_KEY or not TG_BOT_TOKEN:
    print("Помилка: Перевірте файл .env (потрібні YOUTUBE_API_KEY та TELEGRAM_BOT_TOKEN)")
    sys.exit()

bot = TeleBot(TG_BOT_TOKEN)
analyzer = SentimentIntensityAnalyzer()
translator = GoogleTranslator(source='auto', target='en')


# --- ЛОГІКА АНАЛІЗУ (Збережена з минулого разу) ---

def extract_video_id(url):
    if len(url) == 11 and ' ' not in url and '/' not in url: return url
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:8] == '/shorts/': return query.path.split('/')[2]
    return None


def clean_text(text):
    return re.sub(r'(?<=\b\w)\s+(?=\w\b)', '', text)


def analyze_comment(text):
    try:
        translated = translator.translate(text)
    except:
        translated = text
    if not translated: translated = text

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


def get_data(video_id, max_results=30):
    # max_results менше, щоб бот відповідав швидше
    try:
        youtube = build('youtube', 'v3', developerKey=YT_API_KEY)
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=max_results, textFormat="plainText"
        )
        response = request.execute()

        data = []
        for item in response['items']:
            snippet = item['snippet']['topLevelComment']['snippet']
            score, category, trans = analyze_comment(snippet['textDisplay'])
            data.append({
                'Author': snippet['authorDisplayName'],
                'Original': snippet['textDisplay'],
                'Score': score,
                'Category': category
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"API Error: {e}")
        return None


# --- ФУНКЦІЇ ДЛЯ БОТА ---

def generate_report_text(df):
    """Генерує текстове повідомлення зі статистикою"""
    avg_score = df['Score'].mean()
    total = len(df)
    pos = len(df[df['Category'] == 'Positive'])
    neg = len(df[df['Category'] == 'Negative'])

    if avg_score > 0.1:
        verdict = "👍 Позитивний"
    elif avg_score < -0.1:
        verdict = "👎 Негативний"
    else:
        verdict = "😐 Змішаний"

    text = (
        f"📊 <b>Звіт аналізу:</b>\n"
        f"Всього коментарів: {total}\n"
        f"Рейтинг: {avg_score:.2f} (-1..1)\n"
        f"Вердикт: {verdict}\n\n"
        f"💚 Позитивних коментарів: {pos} ({pos / total * 100:.1f}%)\n"
        f"❤️ Негативних коментарів: {neg} ({neg / total * 100:.1f}%)\n"
    )
    return text


def generate_charts(df):
    """Малює графіки і повертає їх як байтовий об'єкт (картинку в пам'яті)"""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))


    counts = df['Category'].value_counts()
    colors = {'Positive': '#66bb6a', 'Neutral': '#fff176', 'Negative': '#ef5350'}
    pie_colors = [colors.get(k, '#bdbdbd') for k in counts.index]
    if len(counts) > 0:
        axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%', colors=pie_colors)
    axes[0].set_title('Емоції')

    # Histogram
    sns.histplot(df['Score'], bins=15, kde=True, ax=axes[1], color='#5c6bc0')
    axes[1].set_title('Розподіл')
    axes[1].axvline(0, color='black', linestyle='--')

    plt.tight_layout()


    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


# --- ОБРОБНИКИ КОМАНД БОТА ---

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привіт! 👋\nНадішли мені посилання на YouTube відео, і я проаналізую коментарі.")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    url = message.text.strip()
    video_id = extract_video_id(url)

    if not video_id:
        bot.reply_to(message, "Це не схоже на посилання YouTube. Спробуй ще раз.")
        return

    # Відправляємо повідомлення
    status_msg = bot.reply_to(message, "⏳ Аналізую коментарі... Це займе хвилину.")

    # Отримуємо дані
    df = get_data(video_id, max_results=40)

    if df is not None and not df.empty:
        # 1. Текстовий звіт
        report = generate_report_text(df)
        bot.send_message(message.chat.id, report, parse_mode='HTML')

        # 2. Графіки
        photo = generate_charts(df)
        bot.send_photo(message.chat.id, photo)

        # 3. CSV файл
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        csv_bytes = io.BytesIO(csv_buffer.getvalue().encode())
        csv_bytes.name = f"report_{video_id}.csv"

        bot.send_document(message.chat.id, csv_bytes, caption="📂 Детальна таблиця")


        bot.delete_message(message.chat.id, status_msg.message_id)

    else:
        bot.edit_message_text("Не вдалося отримати коментарі (або їх немає, або доступ закритий).",
                              message.chat.id, status_msg.message_id)


# --- ЗАПУСК ---
if __name__ == "__main__":
    print("🤖 Бот запущено...")
    bot.polling(none_stop=True)