import nltk
nltk.download('wordnet')         #WordNet synsets and lemmas
nltk.download('stopwords')       #stopwords
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
import pandas as pd

#load data
books = pd.read_csv('data/books.csv')  #read the books file
courses = pd.read_csv('data/courses.csv')  #read the courses file

try:
  books = books.drop(['Unnamed: 5'], axis = 1)
  books = books.drop(['Unnamed: 6'], axis = 1)
except:
  pass

books.head()

#preprocess and combine features
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])  #remove punctuation
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])  #stem and remove stopwords
    return text

#function for content-based recommending
def recommend_books_by_syllabus(course_keywords, books_df, tfidf_matrix):
    course_vec = tfidf.transform([course_keywords])
    similarities = cosine_similarity(course_vec, tfidf_matrix).flatten()
    indices = similarities.argsort()
    recommended_books = books_df.iloc[indices]
    return filter_recommendations(recommended_books, course_keywords)

#filter recommendations for accuracy
def filter_recommendations(recommended_books, course_keywords):
    return recommended_books[recommended_books['title'].str.contains(course_keywords, case=False)]

books['processed_title'] = books['title'].apply(preprocess_text)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['processed_title'])

#example
course_keywords = "java"
recommended_books = recommend_books_by_syllabus(course_keywords, books, tfidf_matrix)
recommended_books = recommended_books.drop(['processed_title'], axis = 1)
print(recommended_books)

books_df = pd.read_csv('data/books.csv')
users_df = pd.read_csv('data/sample user data.csv')  #sample user data

merged_df = users_df.merge(books_df, on='book_id', how='left')
print(merged_df.head())

user_book_matrix = users_df.pivot_table(index='user_id', columns='book_id', aggfunc='size', fill_value=0) #1 signifies book borrowed by user; 0 book not borrowed
print(user_book_matrix.head())

from sklearn.metrics.pairwise import cosine_similarity

#cosine similarity for books
book_similarity = cosine_similarity(user_book_matrix.T)
book_similarity_df = pd.DataFrame(book_similarity, index=user_book_matrix.columns, columns=user_book_matrix.columns)

print(book_similarity_df.head())

import numpy as np

def hybrid_recommend_books(book_title, books_df, tfidf_matrix, tfidf, book_similarity_df, top_n = 5, threshold = 0.2):
    books_copy = books_df.copy()
    #normalise input and dataset
    books_copy['title'] = books_copy['title'].str.strip().str.lower()
    book_title = book_title.strip().lower()
    #check is title is in books_copy
    if book_title not in books_copy['title'].values:
        print(f"Book '{book_title}' not found in dataset.")
        return pd.DataFrame()
    #get index of the book
    book_idx = books_copy[books_copy['title'] == book_title].index[0]

    #content-based similarity
    content_similarities = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix).flatten()

    #collaborative similarity
    collaborative_similarities = np.zeros(len(books_df)) #default to zeros for books with no borrowing data
    book_id = books_copy.loc[book_idx, 'book_id']
    if book_id in book_similarity_df.index:
        collaborative_similarities = book_similarity_df.loc[book_id].reindex(books_copy['book_id'], fill_value=0).values

    #combine similarities with equal weights
    hybrid_similarities = (0.8 * content_similarities + 0.2 * collaborative_similarities) / 2

    #filter low similarity results
    hybrid_similarities[hybrid_similarities < threshold] = 0

    #top n recommendations
    top_indices = np.argsort(hybrid_similarities)[-top_n - 1:][::-1]  #exclude the input book
    top_indices = [idx for idx in top_indices if idx != book_idx][:top_n]

    #metadata for recommended books
    recommendations = books_df.iloc[top_indices].copy()
    recommendations['hybrid_score'] = hybrid_similarities[top_indices]

    req_data = ['book_id', 'title', 'author', 'edition', 'pub_year']
    return recommendations[req_data]

#example
book_title = "Discrete Mathematics – A Concept-based approach"
recommended_books = hybrid_recommend_books(book_title, books_df, tfidf_matrix, tfidf, book_similarity_df)
if not recommended_books.empty:
    print(recommended_books)
else:
    print("No recommendations found.")
