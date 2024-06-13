import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import string

# Ensure NLTK resources are downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Load URLs from Excel file
input_file = 'Input.xlsx'
df = pd.read_excel(input_file)

def extract_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title and article text
        title = soup.find('h1').get_text()
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])
        
        return title + '\n' + article_text
    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return ""

# Create a directory to save extracted texts
if not os.path.exists('extracted_articles'):
    os.makedirs('extracted_articles')

for index, row in df.iterrows():
    url_id = row['URL_ID']
    url = row['URL']
    article_text = extract_article_text(url)
    
    with open(f'extracted_articles/{url_id}.txt', 'w', encoding='utf-8') as file:
        file.write(article_text)

def analyze_text(text):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    # Clean words
    stop_words = set(stopwords.words('english'))
    cleaned_words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
    
    word_count = len(cleaned_words)
    sentence_count = len(sentences)
    syllable_count = sum([len([char for char in word if char in 'aeiouAEIOU']) for word in cleaned_words])
    complex_word_count = sum([1 for word in cleaned_words if syllable_count >= 3])
    avg_sentence_length = word_count / sentence_count if sentence_count else 0
    percentage_complex_words = complex_word_count / word_count if word_count else 0
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    avg_words_per_sentence = word_count / sentence_count if sentence_count else 0
    avg_word_length = sum([len(word) for word in cleaned_words]) / word_count if word_count else 0
    
    personal_pronouns = len([word for word in cleaned_words if word.lower() in ['i', 'we', 'my', 'ours', 'us']])
    
    return {
        'POSITIVE SCORE': sentiment['pos'],
        'NEGATIVE SCORE': sentiment['neg'],
        'POLARITY SCORE': sentiment['compound'],
        'SUBJECTIVITY SCORE': sentiment['neu'],
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllable_count / word_count if word_count else 0,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }

# Load the articles and perform analysis
results = []

for file in os.listdir('extracted_articles'):
    if file.endswith('.txt'):
        url_id = file.replace('.txt', '')
        with open(f'extracted_articles/{file}', 'r', encoding='utf-8') as f:
            text = f.read()
            analysis = analyze_text(text)
            analysis['URL_ID'] = url_id
            analysis['URL'] = df[df['URL_ID'] == url_id]['URL'].values[0]
            results.append(analysis)

# Convert results to DataFrame and save to Excel
output_df = pd.DataFrame(results)
output_df.to_excel('Output Data Structure.xlsx', index=False)
