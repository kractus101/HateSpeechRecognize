import re
import pandas as pd


# data = pd.read_csv("https://raw.githubusercontent.com/gurokeretcha/WishWeightPredictionApplication/master/Fish.csv")
data = pd.read_csv("labeled_data.csv")
# print('before:', data['tweet'])
def remove_special_characters(text):
    
    # Use the sub() function from the re module to remove special characters
    
    text = str(text).lower() # converts text into lower case
    
    text = re.sub(r'@\S+\s', '', text) # removes username of twitter handles
    text = re.sub(r'[^A-Za-z0-9\s]', '', text) # removes special characters
    text = re.sub(r'\d+', '', text) # removes number from the text
    
    return pure_text

   
data['cleaned_tweet'] = data['tweet'].apply(remove_special_characters)
specific_row_value = data['cleaned_tweet'].iloc[24778]
print('last tweet:', specific_row_value)