import pandas as pd

def load_data():
    books = pd.read_csv("dataBooks.csv", delimiter=";", encoding="latin-1", on_bad_lines='skip')
    users = pd.read_csv("data/Users.csv", delimiter=";", encoding="latin-1", on_bad_lines='skip')
    ratings = pd.read_csv("data/Ratings.csv", delimiter=";", encoding="latin-1", on_bad_lines='skip')

    return books, users, ratings

def clean_data(books, users, ratings):
    # Keep only users who rated at least 5 books
    user_rating_counts = ratings['User-ID'].value_counts()
    active_users = user_rating_counts[user_rating_counts >= 5].index
    ratings = ratings[ratings['User-ID'].isin(active_users)]
    
    return books, users, ratings
