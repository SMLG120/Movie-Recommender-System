import re
import csv
import sys

# usage: python parse_logs.py movielog_dump.txt ratings.csv
infile = sys.argv[1]
outfile = sys.argv[2]

rate_re = re.compile(r'^[^,]+,([^,]+),GET /rate/([^=]+)=([1-5])')

with open(infile, 'r', encoding='utf-8') as fin, open(outfile, 'w', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    writer.writerow(['user_id', 'movie_id', 'rating'])
    for line in fin:
        m = rate_re.match(line.strip())
        if m:
            user_id = m.group(1)
            movie_id = m.group(2)
            rating = int(m.group(3))
            writer.writerow([user_id, movie_id, rating])