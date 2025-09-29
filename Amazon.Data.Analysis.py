import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import openpyxl


nltk.download('vader_lexicon') 

#we import our dataset here, ours is amazon reviews
file_path = 'Reviews.csv' 

#is file correct?
try:
    df = pd.read_csv(file_path)
    print("✅ Data load succesfuly")
except FileNotFoundError:
    print(f"❌Error: data {file_path} not founded ")
    exit()

#inter the column
TEXT_COLUMN = 'Text' 

#delete the empty rows
df.dropna(subset=[TEXT_COLUMN], inplace=True) 

#print first data set information
print("\First dataset information")
print(df.head()) 
print(df.info())


#cleaning text for spam comments
def clean_text(text):
    
    text = str(text).lower()
   
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    
    return text

df['cleaned_text'] = df[TEXT_COLUMN].apply(clean_text)

print("\nFirst 5 row after cleaning")
print(df[['cleaned_text']].head())

#start sentiment instensity analyzing
analyzer = SentimentIntensityAnalyzer()

#analyzing sentiment
def analyze_sentiment(text):
    vs = analyzer.polarity_scores(text)
    # if the value is more than 0.05 it's good and if it's under 0.05 it's bad comment
    if vs['compound'] >= 0.05:
        return 'Positive'
    elif vs['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# applying the function in cleaned text
df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)

print("\n show first 5 resualt")
print(df[['cleaned_text', 'sentiment']].head())






#distribution sentiment
sentiment_counts = df['sentiment'].value_counts()
print("\n Distribution sentiment :")
print(sentiment_counts)

# plot the sentiment in circle
plt.figure(figsize=(8, 8))
plt.pie(
    sentiment_counts, 
    labels=sentiment_counts.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=['#4CAF50', '#FF5722', '#FFC107'] # grren for possetive and red for negetive
)
plt.title('(Sentiment Distribution)')
plt.show() 


# rating the sentiment
if 'rating' in df.columns:
    avg_rating_by_sentiment = df.groupby('sentiment')['rating'].mean().sort_values(ascending=False)
    print("\nAverage rating based on emotions:")
    print(avg_rating_by_sentiment)
    
    #Bar chart for comparing scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_rating_by_sentiment.index, y=avg_rating_by_sentiment.values, palette=['#4CAF50', '#FFC107', '#FF5722'])
    plt.title('Average product rating based on analyzed sentiments')
    plt.ylabel('Averege of points')
    plt.xlabel('sentiment')
    plt.show()

   

# --- CODE ADDED FOR EXPORTING RESULTS TO CSV FILE ---

# Define the columns requested: User Name, Product Name, Review Text, Sentiment Analysis
# Column names are assumed based on a typical Amazon review dataset structure.
TEXT_COLUMN = 'Text' 

final_output_columns = []
column_mapping = {}

# 1. User Name: Try ProfileName, then UserId
if 'ProfileName' in df.columns:
    final_output_columns.append('ProfileName')
    column_mapping['ProfileName'] = 'User Name'
elif 'UserId' in df.columns:
    final_output_columns.append('UserId')
    column_mapping['UserId'] = 'User ID'
else:
    print("Warning: User Name column (ProfileName/UserId) not found. Skipping related column.")

# 2. Product Name: ProductId
if 'ProductId' in df.columns:
    final_output_columns.append('ProductId')
    column_mapping['ProductId'] = 'Product ID'
else:
    print("Warning: Product ID column (ProductId) not found. Skipping related column.")

# 3. Original Review Text
if TEXT_COLUMN in df.columns:
    final_output_columns.append(TEXT_COLUMN)
    column_mapping[TEXT_COLUMN] = 'Review Text'

# 4. Sentiment Analysis Result (created in previous steps)
if 'sentiment' in df.columns:
    final_output_columns.append('sentiment')
    column_mapping['sentiment'] = 'Sentiment Analysis'
else:
    print("Error: Sentiment analysis column ('sentiment') not found. Please check previous code execution.")


if final_output_columns:
    # Create the output DataFrame using the selected columns
    df_output = df[final_output_columns].copy()
    
    # Rename columns using the English mapping
    df_output.rename(columns=column_mapping, inplace=True)

    # Save the results to a CSV file with the requested name
    output_filename = 'amazon-comment-result.csv'
    
    # Use encoding='utf-8-sig' to ensure correct display of non-Latin characters (like Persian) in Excel
    try:
        df_output.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n✅ Analysis results successfully saved to '{output_filename}' as a CSV file.")
        print("The file includes User/Product ID, Review Text, and Sentiment Analysis.")
    except Exception as e:
        print(f"\n❌ An error occurred while saving the CSV file: {e}")

else:
    print("\n❌ Saving failed. No valid columns were selected for output.")