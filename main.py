import streamlit as st
from recommendations import get_recommendations

st.title("📚 Book Recommendation System")

# User inputs a book title
book_title = st.text_input("Enter a book title:")

if book_title:
    recommendations = get_recommendations(book_title)
    
    if not recommendations.empty:
        st.write("### Recommended Books:")
        for index, row in recommendations.iterrows():
            st.write(f"**{row['Book-Title']}** by {row['Book-Author']}")
            st.image(row['Image-URL-S'], width=150)
    else:
        st.write("No recommendations found.")
