'''

                                            Movie Recommendation System

  We build a basic content-based recommendation system and collaborative filtering based on user rankings using the MovieLens dataset.

  Movielens dataset: http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

  The dataset contains 100,000 ratings on 9,000 movies by 600 users.
  movies.csv file shows every movie with its id, title, and genre.
  To describe users' relation with movies we used the ratings.csv file which defines every user
with 'userId' and shows their 'rating' based on 'movieId'.
  First, we will use this dataset to recommend movies with a content-based system then we will train our model and
recommend movies with model-based Collaborative Filtering with the KNN algorithm.

  We also used another small dataset to create a GUI for content based sytem. (gui.py)
  IMDB Movie Dataset: https://data.world/promptcloud/imdb-data-from-2006-to-2016


                                                                                                Gizem Kurnaz & Ayşe Ceren Çiçek

'''

# Libraries
import sklearn
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from surprise import SVD
from surprise import accuracy
from fuzzywuzzy import process
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.sparse import csr_matrix
from surprise import Reader, Dataset
from surprise.accuracy import rmse, mae
from sklearn.neighbors import NearestNeighbors
from surprise.model_selection import cross_validate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


                                                # Import data #
movies = pd.read_csv('MovieLensDataset/movies.csv')
ratings = pd.read_csv('MovieLensDataset/ratings.csv')

# Merge two data based on movieId attribute
merged_data = pd.merge(ratings, movies, on='movieId')

# We will not use timestamp attribute so we will drop this column
merged_data.drop(columns=['timestamp'], inplace=True)


movie_mean = pd.DataFrame(merged_data.groupby('title').mean())
movie_mean['counts'] = merged_data.groupby('title')['rating'].count()


# We can plot a histogram to see how rating value distribute
plt.figure(figsize=[10, 6])
movie_mean.rating.hist(bins=50)
plt.xlabel('Rating value')
plt.ylabel('Number of movies')
plt.show(block=False)
plt.savefig('ratingDistribution1.png')
plt.close()




                                                # Data Preprocessing #

df_movies_cnt = pd.DataFrame(ratings.groupby('movieId').size(), columns=['count'])

movie_id_count = pd.DataFrame(merged_data.groupby('movieId').size(), columns=['count'])

# Collect all movie indexes, which has more than 10 votes, in a list called as popular_movies.
popular_movies = list(set(movie_id_count.query('count >= 10').index))

# Check if a movie in merged_data is in populer_movies and stored them in drop_movies.
drop_movies = merged_data[merged_data.movieId.isin(popular_movies)]
print('\nOriginal data: ', merged_data.shape)
print('Data after dropping unpopular movies: ', drop_movies.shape)

# Now we have a movie dataset which includes movies with more than 10 votes.
# We will apply the same process for user who voted less than or equal to 10.
df_users_cnt = pd.DataFrame(drop_movies.groupby('userId').size(), columns=['count'])
active_users = list(set(df_users_cnt.query('count >= 10').index))
drop_users = drop_movies[drop_movies.userId.isin(active_users)]
print('Original data: ', merged_data.shape)
print('Data after dropping both unpopular movies and inactive users: ', drop_users.shape, "\n\n\n")

# Now our dataset can present the real data with less value and more accuracy.
plt.figure(figsize=[10, 6])
drop_users.rating.hist(bins=50)
plt.xlabel('Rating value')
plt.ylabel('Number of movies')
plt.show(block=False)
plt.savefig('ratingDistribution2.png')
plt.close()


# Group movies by their title and count how much they voted.
drop_users.groupby('title')['rating'].count().sort_values(ascending=False).head()

# Create a dataframe to store rating averages of movies and their vote count.
ratings_dataframe = pd.DataFrame(drop_users.groupby('title')['rating'].mean())
ratings_dataframe.rename(columns={'rating': 'avg_rating'}, inplace=True)
ratings_dataframe['votes'] = pd.DataFrame(drop_users.groupby('title')['rating'].count())
ratings_dataframe.sort_values(by='votes', ascending=False).head()





print("\n\n-------------------------------------------------------------------------------------------")


print("                              --- CONTENT BASED FILTERING ---                       \n")

                                                    # Content Based #

# Create a spreadsheet-style pivot table
user_movie_rates = drop_users.pivot_table(index='userId', columns='title', values='rating')

# Select a movie (Zombieland (2009)) to calculate similarity with other movies and improve our model on it.
zombieland_rates = user_movie_rates['Zombieland (2009)']

# Create a profile for each movie based on how the entirety of users rated it.
# Use corrwith() to get correlation between 2 DataFrame objects.
zombieland_corr = user_movie_rates.corrwith(zombieland_rates)

# Add correlation values to the dataframe as a new column.
zombieland_corr = pd.DataFrame(zombieland_corr, columns=['Correlation'])

# Drop NaN values
zombieland_corr.dropna(inplace=True)

# Sort the correlation column in descending order
zombieland_corr.sort_values('Correlation', ascending=False)

# Add the number of votes for those movies
zombieland_corr_count = zombieland_corr.join(ratings_dataframe['votes'])

zombieland_corr_count.sort_values('Correlation', ascending=False)
zombieland_corr_count.sort_values('votes', ascending=False)


# Get movies that have a correlation value higher than 0.4 and has more than 100 votes.
new_zombieland_corr = zombieland_corr_count[zombieland_corr_count['votes'] > 100]
new_zombieland_corr = new_zombieland_corr[new_zombieland_corr['Correlation'] > 0.4]
new_zombieland_corr.sort_values('Correlation', ascending=False)

# Based on the logic we used below, we will create a function to get similar movies of a given movie.


def contentBasedRecommender(movie_name):
    movie_user_ratings = user_movie_rates[movie_name]
    # Create pandas series of correlations for all films with the movie
    movie_corr = user_movie_rates.corrwith(movie_user_ratings)
    # Convert to df
    movie_corr = pd.DataFrame(movie_corr, columns=['Correlation'])
    # Drop nulls
    movie_corr.dropna(inplace=True)
    # Add column for number of votes
    movie_corr = movie_corr.join(ratings_dataframe['votes'])
    # Get movies that have more than 40 votes and the correlation between 0.5 and 1.0
    new_movie_corr = movie_corr[movie_corr['votes'] >= 40]
    new_movie_corr = new_movie_corr[new_movie_corr['Correlation'] > 0.5]
    new_movie_corr = new_movie_corr[new_movie_corr['Correlation'] != 1.0]

    # Sort in descending order
    return new_movie_corr.sort_values('Correlation', ascending=False).head(20)

movie_name = "Fargo (1996)"
print("20 similar movies to ", movie_name, ": \n")
print(contentBasedRecommender(movie_name))


print("\n\n-------------------------------------------------------------------------------------------")


print("                              --- COLLABORATIVE FILTERING with KNN ---                       ")



                                    # User Based Collaborative Filtering With KNN #

# Use sparse matrix to use movieId as rows and userID as columns, and show user ratings for each movie
movie_users = drop_users.pivot(index='movieId', columns='userId', values='rating')

# Replace null values (NaN) with 0
movie_users = movie_users.fillna(0.0)

# Convert to compressed sparse row matrix with csr_matrixmethod
matrix_movie_users = csr_matrix(movie_users.values)

# Use KNN algorithm to train data, and cosine similarity as a distance metric.
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=25)

knn_model.fit(matrix_movie_users)


# Our recommender function will get name of the movie from the user and show 15 neasert neighbors which has least distance.
# We used fuzzywuzzy library for string matching to get better results if user types wrong characters.
# We turn all 15 neighbor's indices and get their titles and show them to user.
# The first index in neighbors is the movie itself and we do not print it.

def movieRecommender(movie_name, data, model, n_recommendations):
    model.fit(data)
    index = process.extractOne(movie_name, movies['title'])[2]
    print('Selected movie is: ', movies['title'][index])
    print('\nRecommendations are:\n')
    distances, indices = model.kneighbors(data[index], n_neighbors=n_recommendations)
    for i in indices[0]:
        if i != index:
            print(movies['title'][i])


movie_name = "toy story"
movieRecommender(movie_name, matrix_movie_users, knn_model, 15 + 1)





print("\n\n-------------------------------------------------------------------------------------------")


print("                              --- MATRIX FACTORIZATION BASED COLLABORATIVE FILTERING WITH SVD METHOD ---                       ")
                        # Matrix Factorization Based Collaborative Filtering with SVD Method #

# Matrix factorization is another class of collaborative filtering algorithms used in recommender systems.
# We will use Singular Value Decomposition method to implement it.

# Data Preprocessing

data = ratings.drop('timestamp', axis=1)


# Check for null values
data.isna().sum()

data.shape

# Check for unique number of movies and users
movie = data['movieId'].nunique()
users = data['userId'].nunique()
print('Number of movies =', movie)
print('Number of users =', users, "\n\n")

# We can see distribution of ratings below (how many times a rating value is given)
data['rating'].value_counts().plot(kind='bar', colormap='terrain')
plt.show(block=False)
plt.savefig('ratingDistribution3.png')
plt.close()


# To get more accurate results, we will only use movies and users which have been rated more than 20 times
filtered_movies = data['movieId'].value_counts() > 20
filtered_movies = filtered_movies[filtered_movies].index.tolist()

filtered_users = data['userId'].value_counts() > 20
filtered_users = filtered_users[filtered_users].index.tolist()

# Now we have filtered users and movies as a list, and we will only use those values on our dataframe
data = data[(data['movieId'].isin(filtered_movies)) & (data['userId'].isin(filtered_users))]


# Now we do not have duplicate data and our data contains 66,405 rows (34,431 rows are removed).


# The Reader class is used to parse a file containing ratings.
reader = Reader(rating_scale=(0.5, 5))

# First create dataset from dataframe. Then create train and test set.
columns = ['userId', 'movieId', 'rating']
dataset = Dataset.load_from_df(data[columns], reader)

trainset = dataset.build_full_trainset()
testset = trainset.build_anti_testset()


# Use SVD method
model = SVD(n_epochs=25, verbose=True)

cross_validate(model, dataset, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print('Training Done\n\n')



# Now we can predict
prediction = model.test(testset)

print("\n         TOP 5 MOVIE RECOMMENDATIONS FOR EACH USER            \n")

# We will define a function to recommend movies based on prediction for each user.
def get_Recommendations(prediction, n):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in prediction:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


top_n = get_Recommendations(prediction, n=5)
for uid, user_ratings in top_n.items():
    movie_id = []
    print("Recommendations for User {}:".format(uid))
    for (iid, ratings) in user_ratings:
        print("\t", movies[movies.movieId == iid]["title"].values[0])
    print("\n")

print("\n         WE MADE 5 MOVIE RECOMMENDATIONS FOR EACH USER WITH MATRIX FACTORIZATION METHOD         \n")

