# 🛡️ AI Content Moderation System (FastAPI + ML)

> A machine learning-powered web application that detects and filters toxic or harmful text in real time using NLP and a deployed API.

---

## 📖 Overview

This project builds an end-to-end AI content moderation system that:

* Trains a machine learning model on toxic comment data
* Deploys the model using FastAPI
* Provides a simple frontend interface for real-time predictions

The system helps automate moderation of user-generated content and reduces manual effort.

---

## 🧠 How It Works

1. User enters text in the frontend
2. Request is sent to FastAPI backend
3. Backend loads trained model (`.pkl`)
4. Text is vectorized using TF-IDF
5. Model predicts whether content is **Toxic / Safe**
6. Result is returned to UI

---

## 🛠️ Tech Stack

* Python
* FastAPI
* Uvicorn
* Scikit-learn
* Pandas, NumPy
* HTML (Frontend)
* Joblib (Model saving/loading)

---

## 📂 Project Structure

```
ai-content-moderation/
│
├── data/
│   └── train.csv
│
├── model/
│   ├── train.py
│   └── moderation_model.pkl
│
├── frontend/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Model Training

* Combined multiple toxicity labels into a **severity score**
* Converted into binary classification:

  * 0 → Safe
  * 1 → Toxic
* Used:

  * TF-IDF Vectorizer
  * Logistic Regression

---

## 🤖 Model Details

* Algorithm: Logistic Regression
* Text Representation: TF-IDF
* Evaluation: Classification Report (Precision, Recall, F1-score)

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone <your-repo-link>
cd ai-content-moderation
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the model (optional)

```bash
python model/train.py
```

### 4️⃣ Run FastAPI server

```bash
uvicorn app:app --reload
```

### 5️⃣ Open in browser

```
http://127.0.0.1:8000
```

---

## 📸 Demo

### 🧾 Enter Text for Moderation
![Input](https://github.com/Kreetikakishore/AI-Content-Moderation/blob/ae33e9d50765522d95d8019b4c4dfaf96046277e/Screenshot%202026-04-03%20205247.png)

### ⚠️ Toxicity Detection Result
![Output](assets/output.png)

---

## 🌍 Real-World Applications

* Social media moderation
* Comment filtering systems
* Online gaming chats
* Community platforms

---

## 💡 Future Improvements

* Use advanced models like BERT
* Add multi-class classification (toxic types)
* Deploy on cloud (AWS / Render / Railway)
* Add user authentication

---

## 📈 Key Impact

✔ Automated detection of harmful content
✔ Reduced manual moderation effort
✔ Real-time prediction using API

---

## 👩‍💻 Author

**Kreetika Kishore**
