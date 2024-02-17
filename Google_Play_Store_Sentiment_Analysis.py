from itertools import count
from nltk.util import pr
import pandas as pd
data = pd.read_csv("user_reviews.csv")
print(data.head())

print(data.isnull().sum())

