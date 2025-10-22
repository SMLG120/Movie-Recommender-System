import pandas as pd
import os
import csv
import re
from collections import defaultdict

# df1 = pd.read_csv('data/raw_data/movies.csv')
# df2 = pd.read_csv('src/data/raw_data/movies.csv')
# df3 = pd.concat([df1,df2]).drop_duplicates()
# print(len(df1),len(df2),len(df3))

# df3.to_csv('data/raw_data/movies.csv',index=False)

# def parse_logs(logfile):
#     """Parse Kafka logs, append only new user-movie pairs, and return new IDs."""
#     print(f"Parsing logs from {logfile}")
#     output_csv = "watch_time.csv"
#     # os.makedirs(os.path.dirname(output_csv), exist_ok=True)

#     existing_pairs = set()
#     if os.path.exists(output_csv):
#         with open(output_csv, "r", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             existing_pairs = {(r["user_id"], r["movie_id"]) for r in reader}

#     user_ids, movie_ids = set(), set()
#     watch_minutes = defaultdict(lambda: defaultdict(set))

#     with open(logfile, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split(",")
#             if len(parts) < 3:
#                 continue
#             user = parts[1]
#             user_ids.add(user)
#             m = re.search(r"/data/m/([^/]+)/(\d+)\.mpg", line)
#             if not m:
#                 continue
#             movie, minute = m.group(1), int(m.group(2))
#             movie_ids.add(movie)
#             watch_minutes[user][movie].add(minute)

#     write_header = not os.path.exists(output_csv)
#     with open(output_csv, "a", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow(["user_id", "movie_id", "interaction_count", "max_minute_reached"])
#         for user, movies in watch_minutes.items():
#             for movie, minutes in movies.items():
#                 if (user, movie) in existing_pairs:
#                     continue
#                 writer.writerow([user, movie, len(minutes), max(minutes)])

#     print(
#         f"Processed {len(user_ids)} users and {len(movie_ids)} movies. \nSaved watch time interaction to {output_csv}"
#     )
#     return user_ids, movie_ids

# parse_logs('data/raw_data/event_stream.log')


# df1 = pd.read_csv('data/raw_data/watch_time.csv')
# df2 = pd.read_csv('watch_time.csv')

# df1.set_index(['user_id','movie_id'], inplace=True)
# df2.set_index(['user_id','movie_id'], inplace=True)
# df1.update(df2[['interaction_count', 'max_minute_reached']])
# df1.reset_index(inplace=True)
# # df1['interaction_count'][(df1['user_id']==df2['user_id']) & (df1['movie_id']==df2['movie_id'])] = df2['interaction_count']
# df1.drop(columns=['minutes_watched'],inplace=True)
# print(df1.isna().sum())
# print(df1.head(10))
# # df3 = pd.concat([df1,df2]).drop_duplicates()
# # print(len(df1),len(df2))
# df1.to_csv('data/raw_data/watch_time.csv',index=False)


# df1 = pd.read_csv('data/raw_data/ratings.csv')
# print(df1.isna().sum())

# import pandas as pd

# df = pd.read_csv("data/raw_data/movies.csv")

# # Compute absolute difference
# df["diff"] = (df["max_minute_reached"] - df["interaction_count"]).abs()

# # Count conditions
# summary = {
#     "equal (diff=0)": (df["diff"] == 0).sum(),
#     "diff = 1": (df["diff"] == 1).sum(),
#     "diff > 1": (df["diff"] > 1).sum(),
#     "total": len(df)
# }

# print(pd.Series(summary))
# print(f"\n% rows with diff > 1: {100 * summary['diff > 1'] / summary['total']:.2f}%")

# print(df[df["diff"] > 1][["user_id", "movie_id", "interaction_count", "max_minute_reached", "diff"]].head(10))
# import glob
# import json

# def flatten_movies_json(movie_dir):
#     """Flatten all movie JSONs in a folder into a single CSV."""
#     movie_csv = "data/raw_data/movies.csv"
#     movies = []

#     for file in glob.glob(f"{movie_dir}/*.json"):
#         with open(file, "r", encoding="utf-8") as f:
#             data = json.load(f)

#             movie = {
#                 "id": data.get("id"),
#                 "title": data.get("title"),
#                 "original_language": data.get("original_language"),
#                 "release_date": data.get("release_date"),
#                 "runtime": data.get("runtime"),
#                 "popularity": data.get("popularity"),
#                 "vote_average": data.get("vote_average"),
#                 "vote_count": data.get("vote_count"),
#                 "genres": ",".join([g.get("name", "") for g in data.get("genres", [])]),
#                 "spoken_languages": ",".join([l.get("name", "") for l in data.get("spoken_languages", [])]),
#                 "production_countries": ",".join([c.get("iso_3166_1", "") for c in data.get("production_countries", [])]),
#                 "overview": (data.get("overview") or "").replace("\n", " ")
#             }
#             movies.append(movie)

#     if not movies:
#         raise ValueError("No movie JSON files found to process.")

#     with open(movie_csv, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=movies[0].keys())
#         writer.writeheader()
#         writer.writerows(movies)

#     print(f"Saved {len(movies)} movies into {movie_csv}")

# flatten_movies_json('data/raw_data/movies')



df = pd.read_csv('data/raw_data/movies.csv')
print(df.isna().sum())