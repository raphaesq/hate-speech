import praw
import csv
from timeit import default_timer as timer

start = timer()

# Login
reddit = praw.Reddit(client_id = "iYA6FvJFQn1Hf9OGaRoz5A",
                    client_secret = "pIfYv3nrbfkbnOVwGoiQQY3f7jMAPA",
                    username = "hatespeechsc",
                    password = "",
                    user_agent = "pythonpraw")

# Gets most downvoted comments in top posts
subreddit = reddit.subreddit('Philippines') # Subreddit
number_of_posts = 10 # Number of posts to go through, set to None if top 1000
top = subreddit.top(params={'t': 'all'}, limit=number_of_posts) # Parameters for which submission's comments, 
file_name = 'RedditFetch-rPhilippines-Top10.csv'
all_comments = []

submission_count = 1
for submission in top: 
    print("Top submission:", submission_count, "of", number_of_posts)
    submission_count += 1
    real_comments = [comment for comment in submission.comments if isinstance(comment, praw.models.Comment)]
    all_comments += real_comments

all_comments.sort(key=lambda comment: comment.score, reverse=False)

most_downvoted_comments = all_comments[:1000000] #bottom 100 comments

# Output to csv
with open(file_name,'w', newline='') as f:
    headers = ['body', 'score']
    writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore', dialect='excel')
    writer.writeheader()

    for post in most_downvoted_comments:
        data = {
            'body' : post.body.encode('utf-8'),
            'score' : post.score
        }

        writer.writerow(data)
        # print(data)

# Timer
elapsed_time = timer() - start
print("Time: ", elapsed_time, "seconds")
