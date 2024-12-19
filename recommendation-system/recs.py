import string
import pandas as pd
import nltk
nltk.download('wordnet')         #WordNet synsets and lemmas
nltk.download('stopwords')       #stopwords
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from rapidfuzz import process, fuzz

#load datasets
books = pd.read_csv("../data/books.csv")
branch_course = pd.read_csv("../data/branch-course.csv")
courses_books = pd.read_csv("../data/courses-books.csv")
courses = pd.read_csv("../data/courses.csv")
user_data = pd.read_csv("../data/sample user data.csv")

#preprocess and combine features
def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])  #remove punctuation
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])  #stem and remove stopwords
    return text

books['processed_title'] = books['title'].apply(preprocess_text)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['processed_title'])

#approximate title matching

def fuzzy_match(book_title, books_df, user_choice, top_n):
    choices = books_df['title'].tolist()
    best_match = process.extractOne(book_title, choices, scorer=fuzz.partial_ratio)
    matched_title = best_match[0]
    if (user_choice == "title"):
        return recommend_books_by_title(matched_title, books_df, tfidf_matrix, top_n)

    if best_match and best_match[1] >= 60: #confidence threshold = 60
        return best_match[0]  # Return the matched title
    else:
        print("No matching book title found.")
        return None
    
#function for title-based recommendations

def recommend_books_by_title(book_title, books_df, tfidf_matrix, top_n=5):
    query_vec = tfidf.transform([book_title])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return books_df.iloc[top_indices][['book_id', 'title', 'author', 'edition', 'pub_year']]

#function for branch & semester based recommendations

def recommend_books_by_branch_semester(branch, semester, branch_course_df, courses_books_df, books_df):
    relevant_courses = branch_course_df[
        (branch_course_df['branch'] == branch) & 
        (branch_course_df['semester'] == semester)
    ]['course_code']

    relevant_books = courses_books_df[
        courses_books_df['course_code'].isin(relevant_courses)
    ]['book_id']

    filtered_books = books_df[
        books_df['book_id'].isin(relevant_books)
    ][['book_id', 'title', 'author', 'edition', 'pub_year']]

    return filtered_books

#hybrid recommendation system (combining title-based and student info-based recommendations)

def hybrid_recommend_books(user_choice, book_title, branch, semester, books_df, branch_course_df, courses_books_df, tfidf_matrix, tfidf, top_n=5):
    #getting book title
    matched_title = fuzzy_match(book_title, books_df, user_choice, top_n)
    if not matched_title:
        return pd.DataFrame()
    
    #data of matched book
    book_idx = books_df[books_df['title'] == matched_title].index
    if book_idx.empty:
        print("Matched title not found in the dataset.")
        return pd.DataFrame()
    book_idx = book_idx[0]

    #title-based filtering
    title_similarities = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix).flatten()

    #student info-based filtering
    relevant_courses = branch_course_df[(branch_course_df['branch'] == branch) & (branch_course_df['semester'] == semester)]['course_code']
    relevant_books = courses_books_df[courses_books_df['course_code'].isin(relevant_courses)]['book_id']

    #combining results
    books_df['title_similarity'] = title_similarities
    books_df['is_relevant'] = books_df['book_id'].isin(relevant_books).astype(int)

    #weighted combination 
    books_df['hybrid_score'] = (0.75 * books_df['title_similarity'] + 0.25 * books_df['is_relevant']) / 2
    top_indices = books_df.sort_values(by='hybrid_score', ascending=False).head(top_n)

    return top_indices[['book_id', 'title', 'author', 'edition', 'pub_year']]

#function to get user input

def recommend_books(user_choice, book_title=None, branch=None, semester=None, books_df=None, branch_course_df=None, 
                    courses_books_df=None, tfidf_matrix=None, tfidf=None, top_n=5):
    if user_choice == "title":
        return fuzzy_match(book_title, books_df, user_choice, top_n)
    elif user_choice == "branch_semester":
        return recommend_books_by_branch_semester(branch, semester, branch_course_df, courses_books_df, books_df)
    elif user_choice == "hybrid":   
        return hybrid_recommend_books(user_choice, book_title, branch, semester, books_df, branch_course_df, courses_books_df, tfidf_matrix, tfidf, top_n)
    else:
        raise ValueError("Invalid choice. Please select 'title', 'branch_semester', or 'hybrid'.")
    
def get_sem(usn):
    if "22" in usn:
        return 5
    elif "23" in usn:
        return 4
    else:
        print("Invalid USN")

def get_branch(usn):
    if "ad" in usn:
        return "AD"
    elif "cs" in usn:
        return "CSE"
    elif "cb" in usn:
        return "CB"
    else:
        print("Invalid USN")

# Example inputs
print("What type of recommendations would you like?")
print("1. From title/topic\t 2. From USN\t 3. Both")
recs = int(input("Enter 1/2/3: "))
if recs == 1:
    user_choice = "title"
    book_title = input("Enter the title or topic: ").lower()
    branch = None
    semester = None
elif recs == 2:
    user_choice = "branch_semester"
    usn = input("Enter your USN: ").lower()
    branch = get_branch(usn)
    semester = get_sem(usn)
    book_title = None
    print(f'Branch: {branch} Semester: {semester}')
elif recs == 3:
    user_choice = "hybrid"
    usn = input("Enter your USN: ").lower()
    branch = get_branch(usn)
    semester = get_sem(usn)
    book_title = input("Enter the title or topic: ").lower()

#recommendations based on user choice
recommended_books = recommend_books(user_choice, book_title=book_title, branch=branch, semester=semester,
                                     books_df=books, branch_course_df=branch_course, 
                                     courses_books_df=courses_books, tfidf_matrix=tfidf_matrix, tfidf=tfidf)
print(recommended_books)