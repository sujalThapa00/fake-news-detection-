import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from imblearn.over_sampling import SMOTE
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import tokenize
import itertools
import string

# Load datasets
fake = pd.read_csv(r"C:\Users\ADMIN\Desktop\Fake.csv")
true = pd.read_csv(r"C:\Users\ADMIN\Desktop\True.csv")

print("\n\n")
print("Shape of true and fake dataframes:")
print(true.shape)
print(fake.shape)

# Add flag to track fake and real
fake['target'] = 'fake'
true['target'] = 'true'

data = pd.concat([fake, true]).reset_index(drop = True)
print("\n")
print("Shape of data:")
print(data.shape)

# Shuffle the data
data = shuffle(data)
data = data.reset_index(drop=True)

# Check the data
print("\n")
print("Data:")
print(data.head())

# Drop date and title columns
data.drop(["date"], axis=1, inplace=True)
data.drop(["title"], axis=1, inplace=True)

# Convert to lowercase
data['text'] = data['text'].apply(lambda x: x.lower())
print("\n")
print("Data (lowercase):")
print(data.head())

# Remove punctuation
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)
print("\n")
print("Data (without punctuation):")
print(data.head())

# Removing stopwords
stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

print("\n\n")
print("Data (without stopwords):")
print(data.head())

"""Basic Data Exploration"""
print(data.groupby(['subject'])['text'].count())
data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()

print(data.groupby(['target'])['text'].count())
data.groupby(['target'])['text'].count().plot(kind="bar")
plt.show()

# Tokenization functions for word frequency analysis
token_space = tokenize.WhitespaceTokenizer()

def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()

# Most frequent words
counter(data[data["target"] == "fake"], "text", 20)
counter(data[data["target"] == "true"], "text", 20)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

"""Preparing The Data"""
# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)

# Convert labels to binary for ROC (fake=0, true=1)
lb = LabelBinarizer()
y_test_binary = lb.fit_transform(y_test)

dct = dict()

"""==================== BASELINE MODELS (No SMOTE) ===================="""

# Vectorize baseline data
vectorizer_baseline = TfidfVectorizer(max_features=5000)
X_train_vec_base = vectorizer_baseline.fit_transform(X_train)
X_test_vec_base = vectorizer_baseline.transform(X_test)

"""Logistic Regression - Baseline"""
print("\n--- Logistic Regression Baseline ---")
lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train_vec_base, y_train)
y_pred_lr_base = lr_baseline.predict(X_test_vec_base)

print("accuracy: {}%".format(round(accuracy_score(y_test, y_pred_lr_base)*100,2)))
dct['Logistic Regression (Baseline)'] = round(accuracy_score(y_test, y_pred_lr_base)*100,2)

# Confusion Matrix: Logistic Regression Baseline
cm_lr_base = confusion_matrix(y_test, y_pred_lr_base)
plot_confusion_matrix(cm_lr_base, classes=['Fake', 'Real'], 
                      title='Confusion matrix - logistic regression base line')

"""Random Forest - Baseline"""
print("\n--- Random Forest Baseline ---")
rf_baseline = RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=42)
rf_baseline.fit(X_train_vec_base, y_train)
y_pred_rf_base = rf_baseline.predict(X_test_vec_base)

print("accuracy: {}%".format(round(accuracy_score(y_test, y_pred_rf_base)*100,2)))
dct['Random Forest (Baseline)'] = round(accuracy_score(y_test, y_pred_rf_base)*100,2)

# Confusion Matrix: Random Forest Baseline
cm_rf_base = confusion_matrix(y_test, y_pred_rf_base)
plot_confusion_matrix(cm_rf_base, classes=['Fake', 'Real'], 
                      title='Confusion matrix - random forest base line')

# Get baseline probability scores for ROC
y_scores_baseline = lr_baseline.predict_proba(X_test_vec_base)[:, 1]
fpr_base, tpr_base, _ = roc_curve(y_test_binary, y_scores_baseline)
roc_auc_base = auc(fpr_base, tpr_base)

"""==================== SMOTE MODELS ===================="""

# Vectorize for SMOTE
vectorizer_smote = TfidfVectorizer(max_features=5000)
X_train_vec_smote = vectorizer_smote.fit_transform(X_train)
X_test_vec_smote = vectorizer_smote.transform(X_test)

print("\nClass distribution before SMOTE:")
print(pd.Series(y_train).value_counts())

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec_smote, y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

"""Logistic Regression - SMOTE"""
print("\n--- Logistic Regression with SMOTE ---")
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_resampled, y_train_resampled)
y_pred_lr_smote = lr_smote.predict(X_test_vec_smote)

print("accuracy: {}%".format(round(accuracy_score(y_test, y_pred_lr_smote)*100,2)))
dct['Logistic Regression (SMOTE)'] = round(accuracy_score(y_test, y_pred_lr_smote)*100,2)

# Confusion Matrix: Logistic Regression with SMOTE (Training Set)
cm_lr_smote_train = confusion_matrix(y_train_resampled, lr_smote.predict(X_train_resampled))
plot_confusion_matrix(cm_lr_smote_train, classes=['Fake', 'Real'], 
                      title='Confusion matrix - logistic regression base line with SMOTE')

# Confusion Matrix: LR with SMOTE (Test Set)
cm_lr_smote_test = confusion_matrix(y_test, y_pred_lr_smote)
plot_confusion_matrix(cm_lr_smote_test, classes=['Fake', 'Real'], 
                      title='Confusion matrix - LR with SMOTE (test set)')

"""Random Forest - SMOTE"""
print("\n--- Random Forest with SMOTE ---")
rf_smote = RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=42)
rf_smote.fit(X_train_resampled, y_train_resampled)
y_pred_rf_smote = rf_smote.predict(X_test_vec_smote)

print("accuracy: {}%".format(round(accuracy_score(y_test, y_pred_rf_smote)*100,2)))
dct['Random Forest (SMOTE)'] = round(accuracy_score(y_test, y_pred_rf_smote)*100,2)

# Confusion Matrix: Random Forest SMOTE
cm_rf_smote = confusion_matrix(y_test, y_pred_rf_smote)
plot_confusion_matrix(cm_rf_smote, classes=['Fake', 'Real'], 
                      title='Confusion matrix - random forest SMOTE')

# Get SMOTE probability scores for ROC
y_scores_smote = lr_smote.predict_proba(X_test_vec_smote)[:, 1]
fpr_smote, tpr_smote, _ = roc_curve(y_test_binary, y_scores_smote)
roc_auc_smote = auc(fpr_smote, tpr_smote)

"""==================== ROC CURVE COMPARISON ===================="""

plt.figure(figsize=(10, 8))

# Plot Baseline
plt.plot(fpr_base, tpr_base, color='blue', linewidth=2, 
         label='Baseline (Test) (AUC = %0.3f)' % roc_auc_base)

# Plot SMOTE
plt.plot(fpr_smote, tpr_smote, color='darkorange', linewidth=2, 
         label='SMOTE (Test) (AUC = %0.3f)' % roc_auc_smote)

# Plot diagonal reference line
plt.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=1.5)

# Labels and title
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - LR Test Set Performance', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print comparison
print(f"\nBaseline AUC: {roc_auc_base:.3f}")
print(f"SMOTE AUC: {roc_auc_smote:.3f}")
print(f"Difference: {abs(roc_auc_base - roc_auc_smote):.3f}")

"""==================== MODEL COMPARISON ===================="""

plt.figure(figsize=(12,7))
plt.bar(list(dct.keys()), list(dct.values()), color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
plt.ylim(90,100)
plt.yticks((91, 92, 93, 94, 95, 96, 97, 98, 99, 100))
plt.title('Model Comparison: Baseline vs SMOTE')
plt.ylabel('Accuracy (%)')
plt.xlabel('Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

print("\n\nFinal Model Comparison:")
for model, acc in dct.items():
    print(f"{model}: {acc}%")

    print("\n================ Fake News Prediction System ================")

while True:
    news = input("\nEnter news text (or type 'exit'): ")
    
    if news.lower() == "exit":
        print("Exiting system...")
        break
    
    # Apply same preprocessing
    news = news.lower()
    news = punctuation_removal(news)
    news = ' '.join([word for word in news.split() if word not in stop])
    
    # Vectorize using SMOTE vectorizer
    news_vec = vectorizer_smote.transform([news])
    
    # Predict using best model (Logistic Regression with SMOTE)
    prediction = lr_smote.predict(news_vec)
    
    if prediction[0] == "fake":
        print("Prediction: FAKE NEWS ")
    else:
        print("Prediction: REAL NEWS ")