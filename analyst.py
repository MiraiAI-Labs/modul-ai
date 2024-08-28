import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from numpy.polynomial.polynomial import Polynomial
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import string
import re
from collections import Counter

def polynomial_regression(series, degree=3):
    x = series.dropna().index.values.astype(float)  # Convert index to float for regression
    y = series.dropna().values  # Get non-NaN values
    p = Polynomial.fit(x, y, degree)  # Fit polynomial regression
    return pd.Series(p(series.index.values.astype(float)), index=series.index)  # Predict using the polynomial model

class Analyzer:
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        self.jobs_data = pd.read_csv(self.csv_dir)
        self.jobs_data_cleaned = self.jobs_data.dropna(subset=['description', 'date_posted'])
        self.jobs_data_cleaned['date_posted'] = pd.to_datetime(self.jobs_data_cleaned['date_posted'], errors='coerce')
        self.all_descriptions = ' '.join(self.jobs_data_cleaned['description'].dropna())
        self.tech_keywords = [
            'python', 'java', 'javascript', 'sql', 'aws', 'docker', 'react', 'angular', 'node', 'linux', 'devops', 'tensorflow', 
            'kubernetes', 'flutter', 'swift', 'cloud', 'ci/cd', 'cybersecurity', 'big data', 'data science', 'machine learning', 
            'deep learning', 'hadoop', 'spark', 'tableau', 'power bi', 'pandas', 'pytorch', 'numpy', 'scikit-learn', 'keras', 
            'figma', 'sketch', 'adobe xd', 'illustrator', 'photoshop', 'android', 'kotlin', 'seo', 'social media', 'marketing', 
            'google ads', 'facebook ads', 'crm', 'content strategy', 'wordpress', 'html', 'css', 'sass'
        ]
        self.jobs_data_cleaned_2023 = self.jobs_data_cleaned[self.jobs_data_cleaned['date_posted'].dt.year >= 2023]
        self.jobs_data_cleaned['is_remote'] = self.jobs_data_cleaned['is_remote'].astype(str).replace('nan', 'no')

    def top_job_titles(self, n=10):
        return dict(self.jobs_data_cleaned['title'].value_counts().head(n))

    def wordcloud(self):
        text = str(self.all_descriptions).lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [word for word in words if len(word) > 2]

        additional_stopwords = {'could', 'would', 'never', 'one', 'even', 'like', 'said', 'say', 'also',
                                'might', 'must', 'every', 'much', 'may', 'two', 'know', 'upon', 'without',
                                'go', 'went', 'got', 'put', 'see', 'seem', 'seemed', 'take', 'taken',
                                'make', 'made', 'come', 'came', 'look', 'looking', 'think', 'thinking',
                                'thought', 'use', 'used', 'find', 'found', 'give', 'given', 'tell', 'told',
                                'ask', 'asked', 'back', 'get', 'getting', 'keep', 'kept', 'let', 'lets',
                                'ensure', 'provide','seems', 'leave', 'left', 'set', 'from', 'subject', 're', 
                                'edu', 'use'}
        
        custom_stopwords = set(stopwords.words('english')).union(additional_stopwords)

        filtered_words = [word for word in words if word not in custom_stopwords]

        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        word_freq = Counter(lemmatized_words)
        sorted_word_freq = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)
        temp_dict = dict()
        for key, val in sorted_word_freq:
            temp_dict[key] = val
        
        return temp_dict
    
    def top10_job_locations(self):
        cleaned_loc = self.jobs_data_cleaned['location'].str.split(',').str[0]
        location_distribution_cleaned = cleaned_loc.value_counts().head(10)
        return dict(location_distribution_cleaned)
    
    def job_posting_trend(self):
        job_posting_trends_2023 = self.jobs_data_cleaned_2023.groupby(self.jobs_data_cleaned_2023['date_posted'].dt.date).size()
        job_posting_trends_2023 = job_posting_trends_2023.rolling(window=7).mean().dropna()
        job_posting_trend_dict = dict()

        for k, v in job_posting_trends_2023.dropna().items():
            job_posting_trend_dict[str(k)] = v

        return job_posting_trend_dict
    
    def top10_industries_with_most_jobs(self):
        industry_distribution = self.jobs_data_cleaned['company_industry'].dropna().value_counts().head(10).sort_values(ascending=True)
        return dict(industry_distribution.sort_index(ascending=True))
    
    def most_mentioned_skills_and_techstacks(self):
        tech_stack_frequency = {keyword: self.all_descriptions.lower().count(keyword) for keyword in self.tech_keywords}
        return tech_stack_frequency
    
    def top10_remote_jobs(self):
        remote_jobs = self.jobs_data_cleaned[self.jobs_data_cleaned['is_remote'] == 'True']
        top_10_remote_job_titles = remote_jobs['title'].value_counts().head(10)     
        top_10_remote_job_titles_sorted = top_10_remote_job_titles.sort_values(ascending=True)  # Sort in ascending order for horizontal barplot
        
        return dict(top_10_remote_job_titles_sorted)
    
    def top10_non_remote_jobs(self):
        non_remote_jobs = self.jobs_data_cleaned[self.jobs_data_cleaned['is_remote'] == 'False']
        top_10_non_remote_job_titles = non_remote_jobs['title'].value_counts().head(10)
        top_10_non_remote_job_titles_sorted = top_10_non_remote_job_titles.sort_values(ascending=True)  # Sort in ascending order for horizontal barplot

        return dict(top_10_non_remote_job_titles_sorted)
    
    def tech_stacks_overtime(self):
        tech_trends_extended = pd.DataFrame(index=self.jobs_data_cleaned_2023['date_posted'].dt.date)
        for keyword in self.tech_keywords:
            tech_trends_extended[keyword] = self.jobs_data_cleaned_2023['description'].str.lower().str.contains(keyword).groupby(self.jobs_data_cleaned_2023['date_posted'].dt.date).sum()

        tech_stack_frequencies = tech_trends_extended.sum().sort_values(ascending=False)
        top_7_tech_stacks = tech_stack_frequencies.head(7).index.tolist()

        tech_trends_extended.index = pd.to_datetime(tech_trends_extended.index)
        tech_trends_extended_numeric = tech_trends_extended.copy()
        tech_trends_extended_numeric.index = tech_trends_extended.index.map(pd.Timestamp.toordinal)

        tech_trends_poly_regression_top7 = tech_trends_extended_numeric[top_7_tech_stacks].apply(polynomial_regression)
        tech_trends_poly_regression_top7.index = pd.to_datetime(tech_trends_poly_regression_top7.index.map(pd.Timestamp.fromordinal))
        tech_trends_poly_regression_top7.index = tech_trends_poly_regression_top7.index.strftime('%Y-%m-%d')
        
        return tech_trends_poly_regression_top7.to_dict()