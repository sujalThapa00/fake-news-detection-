Fake News Detection System

 Project Overview
This project focuses on detecting fake news articles using Machine Learning techniques. 
The model classifies news as **Real** or **Fake** using text feature extraction and supervised learning algorithms.


Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- TF-IDF Vectorization
- SMOTE (for class imbalance handling)
- Matplotlib & Seaborn


## ⚙️ Machine Learning Models
- Logistic Regression
- Random Forest
- Naive Bayes
- Support Vector Machine (SVM)


Workflow
1. Data Cleaning & Preprocessing  
2. TF-IDF Feature Extraction  
3. Train-Test Split  
4. Class Imbalance Handling using SMOTE  
5. Model Training  
6. Evaluation using:
   - Confusion Matrix
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - ROC Curve


 Evaluation Metrics
Performance was evaluated using:
- Confusion Matrix
- Precision & Recall
- F1 Score
- ROC-AUC Curve

 Results
The model achieved strong classification performance with improved minority class detection after applying SMOTE.

How to Run
```bash
pip install -r requirements.txt
python main.py
