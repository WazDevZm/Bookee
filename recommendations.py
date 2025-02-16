import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

from data_processing import load_data

def train_model():
    books, users, ratings = load_data()
    
    # Convert ratings into Surprise format
    reader = Reader(rating_scale=(0, 10))
    data = Dataset.load_from_df(ratings[['User-ID', 'ISBN', 'Book-Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    
    # Train model
    model = SVD()
    model.fit(trainset)

    # Save model
    with open("models/svd_model.pkl", "wb") as f:
        pickle.dump(model, f)

def get_recommendations(book_title):
    books, _, ratings = load_data()
    
    # Find the book's ISBN
    isbn = books.loc[books['Book-Title'].str.contains(book_title, case=False, na=False), 'ISBN'].values[0]

    # Load model
    with open("models/svd_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict ratings for all books
    book_preds = []
    for book in books['ISBN'].unique():
        pred = model.predict(uid=99999, iid=book).est
        book_preds.append((book, pred))

    # Sort by highest rating
    book_preds.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 5 recommendations
    top_books = books[books['ISBN'].isin([b[0] for b in book_preds[:5]])]
    
    return top_books
