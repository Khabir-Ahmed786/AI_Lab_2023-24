# Ex: 13 AI MINI PROJECT - TWITTER SENTIMENT ANALYSIS FOR ELECTRONIC PRODUCT SALE 
### DATE:   04/11/24                                                         
### REGISTER NUMBER : 212221040083
### AIM: 
To develop a machine learning model to analyze public sentiment on Twitter regarding a specific electronic product sale. By gathering and analyzing tweets.

###  Algorithm:


### The working of the Random Forest algorithm in steps:

Step-1: Select random K data points from the training set.  
Step-2: Build the decision trees associated with the selected data points (Subsets).  
Step-3: Choose the number N for decision trees that you want to build.  
Step-4: Repeat Step 1 & 2.  
Step-5: For new data points, find the predictions of each decision tree, and assign the new data 
points to the category that wins the majority votes.



### Working of Support vector machine:
  
Step-1: Data of finite-dimensional space is mapped to p-dimension and aims at finding p-1 
dimension.  
Step-2: Creates two parrel hyperplanes on either side that passes through the nearest data points.  
Step-3:  The region bound by these two hyperplanes is margin.  
Step-4: Hyperplanes drawn to classify the data.  
Step-5: Most stable hyperplane is maximum margin. Margin is distance between two classes. 



### Working of Decision tree:  

Step-1: Begin the tree with the root node, say S which contains the complete dataset.  
Step-2: Find the best attribute in the dataset using Attribute Selection Measure.  
Step-3: Divide the S into subsets that contains possible values for the best attributes.  
Step-4: Generate the decision tree node, which contains the best attribute.  
Step-5: repeat the step 3 until you get a final node.




### Program:

```
import numpy as np import pandas as pd import seaborn as sns import 
matplotlib.pyplot as plt import nltk 
nltk.download('averaged_perceptron_tagger') nltk.download('vader_lexicon') 
from nltk.corpus import stopwords from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize import re from sklearn.naive_bayes import 
GaussianNB from sklearn.ensemble import RandomForestClassifier from 
sklearn.metrics import accuracy_score from nltk.sentiment.vader import 
SentimentIntensityAnalyzer from sklearn.svm import LinearSVC from 
sklearn.feature_extraction.text import CountVectorizer, TfidfVectoriz er 
from sklearn.linear_model import SGDClassifier from 
sklearn.model_selection import cross_val_score, train_test_split from 
sklearn.metrics import f1_score, accuracy_score import warnings 
warnings.filterwarnings('ignore') 
pd.set_option('display.notebook_repr_html', True) train = 
pd.read_csv('train.csv', encoding= 'unicode_escape') test = 
pd.read_csv('test.csv', encoding= 'unicode_escape') df = pd.concat([train, 
test]) df.head()  train.head() test.head() print(train.shape, test.shape, 
df.shape) df.describe() df.dtypes df.isnull().sum() 
df['label'].value_counts() sns.countplot(x='label', data=df)  
  
# Cleaning Raw tweets def 
clean_text(text):  
      
    #remove emails     text = ' '.join([i for i in 
text.split() if '@' not in i])  
      
    #remove web address     text = 
re.sub('http[s]?://\S+', '', text)  
      
    #Filter to allow only alphabets     text 
= re.sub(r'[^a-zA-Z\']', ' ', text)  
      
    #Remove Unicode characters     text = 
re.sub(r'[^\x00-\x7F]+', '', text)  
      
    #Convert to lowercase to maintain consistency     
text = text.lower()  
      
 
 
    #remove double spaces      
text = re.sub('\s+', ' ',text)  
         return 
text  
 df["clean_tweet"] = df.tweet.apply(lambda x: 
clean_text(x))  
#defining stop words  
STOP_WORDS = ['a', 'about', 'above', 'after', 'again', 'against', 'all', ' 
also', 'am', 'an', 'and',  
              'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been',  
'before', 'being', 'below',  
              'between', 'both', 'but', 'by', 'can', "can't", 'cannot', 'c 
om', 'could', "couldn't", 'did',  
              "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down',  
'during', 'each', 'else', 'ever',  
              'few', 'for', 'from', 'further', 'get', 'had', "hadn't", 'ha 
s', "hasn't", 'have', "haven't", 'having',  
              'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'her 
s', 'herself', 'him', 'himself', 'his', 'how',  
              "how's", 'however', 'http', 'i', "i'd", "i'll", "i'm", "i've 
", 'if', 'in', 'into', 'is', "isn't", 'it',  
              "it's", 'its', 'itself', 'just', 'k', "let's", 'like', 'me',  
'more', 'most', "mustn't", 'my', 'myself',  
              'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or',  
'other', 'otherwise', 'ought', 'our', 'ours',  
              'ourselves', 'out', 'over', 'own', 'r', 'same', 'shall', "sh 
an't", 'she', "she'd", "she'll", "she's",  
              'should', "shouldn't", 'since', 'so', 'some', 'such', 'than' 
, 'that', "that's", 'the', 'their', 'theirs',  
              'them', 'themselves', 'then', 'there', "there's", 'these', ' 
they', "they'd", "they'll", "they're",  
              "they've", 'this', 'those', 'through', 'to', 'too', 'under', 
 'until', 'up', 'very', 'was', "wasn't",  
              'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 
'what', "what's", 'when', "when's", 'where',  
              "where's", 'which', 'while', 'who', "who's", 'whom', 'why',  
"why's", 'with', "won't", 'would', "wouldn't",  
              'www', 'you', "you'd", "you'll", "you're", "you've", 'your', 
 'yours', 'yourself', 'yourselves'] # Remove stopwords from all the tweets 
df['cleaned_tweet'] = df['clean_tweet'].apply(lambda x: ' '.join([word for  
word in x.split() if word not in (STOP_WORDS)]))  
#Adding New feature length of Tweet 
df['word_count']=df.cleaned_tweet.str.split().apply(lambda x: len(x))  
#Adding New Feature Polarity Score sid= SentimentIntensityAnalyzer() 
sid.polarity_scores(df.iloc[0]['cleaned_tweet']) df['scores'] 
 
 
=df['tweet'].apply(lambda tweet: sid.polarity_scores(tweet)) 
df['compound'] =df['scores'].apply(lambda d:d['compound']) 
df['comp_score'] = df['compound'].apply(lambda score: '0' if score>=0 else  
'1')  
  
# Remove unnecessary ndf=df.copy() ndf = 
ndf.drop(['tweet','clean_tweet','scores','compound','word_count','co 
mp_score'], axis = 1) ndf.head()  
# Seperating Train and Test Set train_set 
= ndf[~ndf.label.isnull()] test_set = 
ndf[ndf.label.isnull()]  
# Shape  
print(train_set.shape,test_set.shape) # 
Defining X and Y  
X = train_set.drop(['label'], axis=1) 
y = train_set.label # Dropping target 
columns  
test_set = test_set.drop(['label'], axis=1)  
X=X['cleaned_tweet'].astype(str)  
#Train test Split  
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, 
random_state = 3) print(X_train.shape, X_test.shape, y_train.shape, 
y_test.shape)  
  
#Random Forest  
 from sklearn.feature_extraction.text import 
TfidfTransformer vect = CountVectorizer() vect.fit(X_train)  
X_train_dtm = vect.transform(X_train) X_test_dtm = 
vect.transform(X_test) model = 
RandomForestClassifier(n_estimators=200) 
model.fit(X_train_dtm,y_train) rf = 
model.predict(X_test_dtm) 
print("Accuracy:",accuracy_score(y_test,rf)*100,"%") 
print("Precision:",metrics.precision_score(y_test,y_pred)) 
print("Recall:",metrics.recall_score(y_test,y_pred))  
  
  
#SVM(Support vector classifier)  
 from sklearn import svm clf=svm.SVC(kernel='linear') 
clf.fit(X_train_dtm,y_train) 
y_pred=clf.predict(X_test_dtm) from sklearn import metrics 
print("Accuracy:",metrics.accuracy_score(y_test,y_pred)) 
print("Recall:",metrics.precision_score(y_test,y_pred)) 
print("Precision:",metrics.precision_score(y_test,y_pred))  
  
 
 
#Decision Tree(Decision tree classifier)  
  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, r 
andom_state=1) from sklearn.tree import DecisionTreeClassifier 
dt=DecisionTreeClassifier() dt.fit(X_train_dtm,y_train) from sklearn 
import metrics  
print("Accuracy:",metrics.accuracy_score(y_test,y_pred)) 
print("Precision:",metrics.precision_score(y_test,y_pred)) 
print("Recall:",metrics.recall_score(y_test,y_pred))  
  import matplotlib.pyplot as 
plt  
 activities = ['Random Forest classifier','Support vector 
classifier','Deci sion tree classifier']  
  
# portion covered by each label slices 
= [0.72302,0.72302,0.2592]  
  
# color for each label colors 
= ['r', 'y', 'g']  
  
# plotting the pie chart plt.pie(slices, 
labels=activities, colors=colors,         
startangle=90, shadow=True, explode=(0, 0, 0),         
radius=1.2, autopct='%1.1f%%') 
plt.title('Precision\n') plt.show()  
  import matplotlib.pyplot as 
plt  
 activities = ['Random Forest classifier','Support vector 
classifier','Deci sion tree classifier']  
  
# portion covered by each label slices 
= [0.7348,0.73486,0.25354]   
# color for each label colors 
= ['b', 'r', 'y']   
# plotting the pie chart plt.pie(slices, 
labels=activities, colors=colors,         
startangle=90, shadow=True, explode=(0, 0, 0),         
radius=1.2, autopct='%1.1f%%') plt.title('Recall\n') 
plt.show()  
   
names=['RF','SVM','DT'] 
values=[86.195,85.94,60.690] plt.bar(names,values)   

```

### Output:

![Screenshot 2024-11-12 202542](https://github.com/user-attachments/assets/febcdec8-d156-40f1-8575-f05d6e5286c9)

![Screenshot 2024-11-12 202616](https://github.com/user-attachments/assets/0a30edf3-3c3b-46a5-ba4b-3697b9c7cc65)

![Screenshot 2024-11-12 202659](https://github.com/user-attachments/assets/7ef0919c-3caa-48d9-9e5f-40019055118a)

![Screenshot 2024-11-12 202804](https://github.com/user-attachments/assets/a6c557e1-5a3f-4cf9-b8bd-cebd9fca1dba)







### Result:
The model achieved a satisfactory level of accuracy, demonstrating its capability in predicting the specified target outcome.
