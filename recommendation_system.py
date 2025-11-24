from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# 1. MOVIE DATA
# -------------------------
movies = [
    {"title": "Inception", "description": "A thief who steals corporate secrets using dream-sharing technology."},
    {"title": "Interstellar", "description": "A team travels through a wormhole in space to ensure humanity's survival."},
    {"title": "The Dark Knight", "description": "Batman faces the Joker in Gotham City."},
    {"title": "Shutter Island", "description": "A U.S. Marshal investigates a psychiatric facility on a remote island."},
    {"title": "The Martian", "description": "An astronaut becomes stranded on Mars and must survive alone."},
    {"title": "Gravity", "description": "Two astronauts are stranded in space after an accident."}
]

# Extract titles & descriptions
movie_names = [m["title"] for m in movies]
descriptions = [m["description"] for m in movies]

# Create lowercase → original name map
lower_name_map = {title.lower(): title for title in movie_names}

# -------------------------
# 2. TF–IDF MODEL
# -------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(descriptions)
similarity_matrix = cosine_similarity(tfidf_matrix)

# -------------------------
# 3. RECOMMENDATION FUNCTION
# -------------------------
def recommend(movie_name, top_n=3):
    movie_name = movie_name.lower()

    if movie_name not in lower_name_map:
        return None  # movie not found

    original_name = lower_name_map[movie_name]

    index = movie_names.index(original_name)
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommended = []
    for i, score in similarity_scores[1:top_n + 1]:
        recommended.append(movie_names[i])  # preserve original names

    return recommended

# -------------------------
# 4. USER INTERFACE LOOP
# -------------------------
print("🎬 Movie Recommendation System")
print("Type a movie name (or type 'exit' to quit)\n")

while True:
    user_input = input("Enter movie name: ").strip()

    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    results = recommend(user_input)

    if results is None:
        print("❌ Movie not found. Please try again.\n")
    else:
        print("\nRecommended Movies:")
        for r in results:
            print("👉", r)
        print()
