import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load("pkl/spam_classifier_model.pkl")
vectorizer = joblib.load("pkl/tfidf_vectorizer.pkl")

# Get class labels
classes = model.classes_
label_map = {0: "HAM", 1: "SPAM"}
feature_names = vectorizer.get_feature_names_out()

# Set up subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("🔍 Top 10 Words by Class", fontsize=16, weight='bold')

# Loop through each class (HAM & SPAM)
for i, label in enumerate(classes):
    # Get top 10 feature indices for this class
    top_indices = np.argsort(model.feature_log_prob_[i])[::-1][:10]
    top_words = [feature_names[j] for j in top_indices]
    top_probs = model.feature_log_prob_[i][top_indices]

    # Plot
    axes[i].barh(top_words[::-1], top_probs[::-1], color='orange' if label == 1 else 'skyblue')
    axes[i].set_title(f"{label_map[label]} 📩")
    axes[i].set_xlabel("Log Probability")
    axes[i].tick_params(axis='y', labelsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
