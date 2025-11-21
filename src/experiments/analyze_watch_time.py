import pandas as pd
import os

# Read the CSV file
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'raw_data', 'watch_time.csv')
df = pd.read_csv(data_path)

# Group by movie_id and sum the minutes_watched
popular_movies = df.groupby('movie_id')['minutes_watched'].sum().reset_index()

# Sort by total minutes watched in descending order
popular_movies = popular_movies.sort_values(by='minutes_watched', ascending=False)

# Calculate additional statistics
total_watch_time = popular_movies['minutes_watched'].sum()
average_watch_time = popular_movies['minutes_watched'].mean()
median_watch_time = popular_movies['minutes_watched'].median()
std_watch_time = popular_movies['minutes_watched'].std()
num_movies = len(popular_movies)

print("Top 10 Most Popular Movies by Total Watch Time:")
print(popular_movies.head(10))

print(f"\nTotal watch time across all movies: {total_watch_time} minutes")
print(f"Average watch time per movie: {average_watch_time:.2f} minutes")
print(f"Median watch time per movie: {median_watch_time:.2f} minutes")
print(f"Standard deviation of watch time: {std_watch_time:.2f} minutes")
print(f"Number of movies: {num_movies}")

# Optionally, save to a new CSV
output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'popular_movies.csv')
popular_movies.to_csv(output_path, index=False)
