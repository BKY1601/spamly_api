# 🚀 Spamly API – SMS Spam Classifier as a RESTful API

**Spamly API** is a RESTful API built using **FastAPI** that allows users to classify SMS messages as **spam** or **ham** using a trained machine learning model. This project is an extension of the original **Spamly** project and makes the classifier accessible via HTTP endpoints for integration into other applications.

---

## 📌 Features

- Accepts SMS messages via POST request
- Returns prediction: `spam` or `ham`
- Built using **FastAPI**
- Loads a pre-trained **Scikit-learn model**
- Provides interactive Swagger UI for testing

---

## 🛠 Tech Stack

- Python  
- FastAPI  
- Scikit-learn  
- Pandas  
- Uvicorn  
- Pickle (for loading the model)

---

## 📁 Project Structure

`````
├── model/ #Model script
├── pkl/ saved pickel files from model 
│ ├── spam_model.pkl # Trained model
│ └── vectorizer.pkl # TF-IDF vectorizer
├── main.py # FastAPI application
├── requirements.txt # Dependencies
└── README.md # Project documentation
``````

---

## ▶️ How to Run the Project

1. **Clone the repository**
  bash
  git clone https://github.com/yourusername/spamly-api.git
  cd spamly-api

2. **Install Dependencies**
  pip install -r requirements.txt
   
3. **Runi the FastAPI server**
  uvicorn main:app --reload

4. **Open in browser**
  http://127.0.0.1:8000/docs – Swagger UI
  http://127.0.0.1:8000/redoc – ReDoc UI

📬 API Endpoint
POST /predict
Request Example (JSON)

{
  "message": "Congratulations! You have won a free ticket."
}

Response Example 

{
  "prediction": "spam"
}

-----

📈 Future Enhancements
Add user authentication

Deploy to cloud (Render, Railway, etc.)

Add logging and error handling

Build a frontend to consume the API

---

## 👨‍💻 Author

**Bipin Yadav**  
📧 bipinyadav919@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/bipin-yadav-jan16)  
🔗 [GitHub](https://github.com/BKY1601)
