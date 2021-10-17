import pymongo
import os
from dotenv import load_dotenv
load_dotenv()


connection=os.getenv('connection')
conn=pymongo.Client(connection)
