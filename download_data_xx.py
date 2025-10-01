import re
import csv
from collections import defaultdict

def parse_watch_logs(logfile):
    user_ids = set()
    movie_ids = set()
    watch_time = defaultdict(lambda: defaultdict(int))  # watch_time[user][movie] = minutes watched

    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue

            user = parts[1]
            user_ids.add(user)

            m = re.search(r"/data/m/([^/]+)/(\d+)\.mpg", line)
            if m:
                movie = m.group(1)  # e.g"shrek+2+2004"
                minute = int(m.group(2))
                movie_ids.add(movie)
                watch_time[user][movie] += 1  # count 1-minute chunks

    return user_ids, movie_ids, watch_time

# user_ids, movie_ids, watch_time = parse_watch_logs("data/raw_data/event_stream.log")
# print("Unique users:", len(user_ids))
# print("Unique movies:", len(movie_ids))
# print("Sample watch time:", dict(list(watch_time.items())[:2]))
import pandas as pd
data = pd.read_csv("data/training_data.csv")
print([x for x in data['gender'].unique()])

# from confluent_kafka import Consumer

# c = Consumer({
#     'bootstrap.servers': 'fall2025-comp585.cs.mcgill.ca:9092',
#     'group.id': 'recsys',
#     'auto.offset.reset': 'earliest'
# })
# c.subscribe(['movielog6'])

# user_ids = set(open("user_ids.txt").read().splitlines())
# movie_ids = set(open("movie_ids.txt").read().splitlines())

# while True:
#     msg = c.poll(1.0)
#     if msg is None: 
#         continue
#     line = msg.value().decode('utf-8')

#     if "/rate/" in line:
#         parts = line.split(",")
#         user_id = parts[1]
#         if user_id in user_ids:
#             print(line)  # or write to file


# with open("user_ids.txt", "w") as uf:
#     uf.write("\n".join(user_ids))

# with open("movie_ids.txt", "w") as mf:
#     mf.write("\n".join(movie_ids))

# with open("watch_time.csv", "w", newline="") as cf:
#     writer = csv.writer(cf)
#     writer.writerow(["user_id", "movie_id", "minutes_watched"])
#     for u, movies in watch_time.items():
#         for m, minutes in movies.items():
#             writer.writerow([u, m, minutes])


# def load_ids(path):
#     with open(path) as f:
#         return set(line.strip() for line in f if line.strip())

# # Load both versions
# old_users = load_ids("data/user_ids.txt")
# new_users = load_ids("user_ids.txt")
# old_movies = load_ids("data/movie_ids.txt")
# new_movies = load_ids("movie_ids.txt")

# # Compute overlap
# user_overlap = old_users & new_users
# movie_overlap = old_movies & new_movies

# print(f"Users: old={len(old_users)}, new={len(new_users)}, overlap={len(user_overlap)}")
# print(f"Movies: old={len(old_movies)}, new={len(new_movies)}, overlap={len(movie_overlap)}")

# # Optional: percentage overlap
# print(f"User overlap: {len(user_overlap)/len(old_users)*100:.2f}% of old, "
#       f"{len(user_overlap)/len(new_users)*100:.2f}% of new")
# print(f"Movie overlap: {len(movie_overlap)/len(old_movies)*100:.2f}% of old, "
#       f"{len(movie_overlap)/len(new_movies)*100:.2f}% of new")


# def save_extra_ids(old_file, new_file, output_file):
#     with open(old_file) as f:
#         old_ids = set(line.strip() for line in f if line.strip())
#     with open(new_file) as f:
#         new_ids = set(line.strip() for line in f if line.strip())

#     extras = new_ids - old_ids
#     with open(output_file, "w") as f:
#         f.write("\n".join(sorted(extras)))

#     print(f"[INFO] Saved {len(extras)} extras into {output_file}")
#     return extras

# # Save extra users and movies
# extra_users = save_extra_ids("data/user_ids.txt", "user_ids.txt", "data/user_test_ids.txt")
# extra_movies = save_extra_ids("data/movie_ids.txt", "movie_ids.txt", "data/movie_test_ids.txt")

