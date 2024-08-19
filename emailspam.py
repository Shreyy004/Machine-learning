import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re  # regular expression
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Read the dataset
messages = pd.read_csv('SMSSpamCollection.tsv', sep='\t', names=['LABEL', 'MESSAGES'])

# Plot the count of ham and spam messages
sns.countplot(x='LABEL', data=messages)
plt.show()

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Preprocess the messages
corpus = []
for i in range(0, len(messages)):
    # Remove non-alphabetic characters
    review = re.sub('[^a-zA-Z]', ' ', messages['MESSAGES'][i])
    # Convert to lowercase
    review = review.lower()
    # Tokenize the text into words
    review = review.split()
    # Apply stemming and remove stopwords
    review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
    # Join the processed words back into a string
    review = ' '.join(review)
    corpus.append(review)

print(corpus)

# Create the Bag of Words model
cv = CountVectorizer(max_features=3500)
X = cv.fit_transform(corpus).toarray()
y = pd.get_dummies(messages['LABEL']).iloc[:, 1].values

print("X:", X)
print("y:", y)

# Create a pickle file for the CountVectorizer
with open('cv_transform.pkl', 'wb') as cv_file:
    pickle.dump(cv, cv_file)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train:", X_train)
print("X_test:", X_test)

# Train the Multinomial Naive Bayes model
mnb = MultinomialNB(alpha=0.8)
mnb.fit(X_train, y_train)

# Predict and evaluate the model
y_pred_mnb = mnb.predict(X_test)
mnb_acc = accuracy_score(y_pred_mnb, y_test)
print("MNB Accuracy:", mnb_acc)

# Model prediction example
message = 'Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...'
data = [message]
vect = cv.transform(data).toarray()
my_prediction = mnb.predict(vect)
if my_prediction == 0:
    print("It's a Ham Mail")
else:
    print("It's a Spam Mail")

# Create a pickle file for the Multinomial Naive Bayes model
with open('model.pkl', 'wb') as model_file:
    pickle.dump(mnb, model_file)
