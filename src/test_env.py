from dotenv import load_dotenv
import os

load_dotenv()

print("MONGO:", os.getenv("MONGO_URI"))
