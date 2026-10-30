import joblib
import sys

# Load trained model and TF-IDF vectorizer
model = joblib.load("category_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


def predict_category(product_title):
    product_title = product_title.lower().strip()

    title_vector = tfidf.transform([product_title])

    prediction = model.predict(title_vector)

    return prediction[0]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        title = " ".join(sys.argv[1:])
    else:
        title = input("Enter product title: ")

    category = predict_category(title)

    print("Predicted category:", category)
