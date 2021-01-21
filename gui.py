
                                    # Content Based With Another Dataset #

# We developed a basic GUI with a small dataset (IMDB movie dataset) which has more attributes.


# Libraries
from tkinter import *
import string
import pandas as pd
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import process
warnings.filterwarnings("ignore", category=UserWarning)

# Create a window
window = Tk()
window.geometry("700x550")

# Read csv file
df = pd.read_csv("IMDB_Dataset/movie_data.csv")


# Function to get columns that we will use, from data
def get_important_features(data):
    important_features = []
    for i in range(0, data.shape[0]):
        lis = [str(data['movie_title'][i]), str(data["actor_1_name"][i]), str(data['actor_2_name'][i]),str(data['director_name'][i]), str(data['genres'][i])]
        x = ' '.join(lis)
        important_features.append(x)

    return important_features


df["important_features"] = get_important_features(df)

cm = CountVectorizer().fit_transform(df["important_features"])

# Calculate cosine similarity
cs = cosine_similarity(cm)

# Function to turn top similar 10 movies
def recommender(movie_name):

    movies = ''

    # Using fuzzy matching to get most correct title of movie even if user types wrong or in lower case
    index = process.extractOne(movie_name, df["movie_title"])[0]

    # Get the index of the movie which user typed
    movie_id = df[df.movie_title == index]["movie_id"].values[0]

    # Get the cosine similarity of this movie with others as list
    scores = list(enumerate(cs[movie_id]))

    # Sort score list so we can get most similar ones
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_scores = sorted_scores[1:]

    j = 0

    for item in sorted_scores:

        # For every movie in list get title from its id
        movie_title = df[df.movie_id == item[0]]["movie_title"].values[0]

        # Concatenate movie names and its order
        text = str(j+1) + '  ' + movie_title + '\n\n'

        # Store movie names in string to print easily in label
        movies = movies + text

        j = j+1

        if j > 9:
            # If we get 10 movies we will stop here
            break

    # Print movie names in label3
    label3.config(text=movies, font=("Arial", 14))


# Function to get entered movie name and call recommender function to show recommendations
def rec_movie():
    movie = entry.get()
    label2.config(text="The most recommended movies to '{0}' are:".format(string.capwords(movie)), fg="wheat4")
    recommender(movie)


# Set title to gui window
window.title("Movie Recommendation")

label = Label(window, text="Write a movie to get recommendations:", font=("Arial", 18), fg="wheat4")
label.place(x=40, y=20)

label2 = Label(window, text="", font=("Arial", 18))
label2.place(x=40, y=150)

entry = Entry(window)
entry.place(x=200, y=70)

button = Button(window, text="Ok", command=rec_movie)
button.place(x=280, y=100)

label3 = Label(window, anchor="e", justify=LEFT)
label3.place(x=40, y=200)

mainloop()
