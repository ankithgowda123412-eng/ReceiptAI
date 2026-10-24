
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


training_data = [
    # Groceries
    ("rice", "Groceries"),
    ("wheat", "Groceries"),
    ("dal", "Groceries"),
    ("sugar", "Groceries"),
    ("salt", "Groceries"),
    ("milk", "Groceries"),
    ("vegetables", "Groceries"),
    ("fruits", "Groceries"),
    ("bread", "Groceries"),
    ("flour", "Groceries"),

    # Personal Care
    ("shampoo", "Personal Care"),
    ("soap", "Personal Care"),
    ("toothpaste", "Personal Care"),
    ("face wash", "Personal Care"),
    ("conditioner", "Personal Care"),
    ("cream", "Personal Care"),
    ("deodorant", "Personal Care"),

    # Healthcare
    ("paracetamol", "Healthcare"),
    ("medicine", "Healthcare"),
    ("tablet", "Healthcare"),
    ("syrup", "Healthcare"),
    ("vitamin", "Healthcare"),
    ("pharmacy", "Healthcare"),

    # Clothing
    ("shirt", "Clothing"),
    ("pant", "Clothing"),
    ("jeans", "Clothing"),
    ("t shirt", "Clothing"),
    ("dress", "Clothing"),
    ("jacket", "Clothing"),

    # Electronics
    ("laptop", "Electronics"),
    ("computer", "Electronics"),
    ("keyboard", "Electronics"),
    ("mouse", "Electronics"),
    ("mobile", "Electronics"),
    ("phone", "Electronics"),
    ("charger", "Electronics"),
    ("headphones", "Electronics"),

    # Transportation
    ("petrol", "Transportation"),
    ("diesel", "Transportation"),
    ("bus ticket", "Transportation"),
    ("taxi", "Transportation"),
    ("uber", "Transportation"),
    ("fuel", "Transportation"),

    # Education
    ("notebook", "Education"),
    ("book", "Education"),
    ("pen", "Education"),
    ("pencil", "Education"),
    ("school", "Education"),
    ("college", "Education"),

    # Household
    ("chair", "Household"),
    ("table", "Household"),
    ("bucket", "Household"),
    ("detergent", "Household"),
    ("cleaner", "Household"),

    # Food
    ("pizza", "Food"),
    ("burger", "Food"),
    ("restaurant", "Food"),
    ("coffee", "Food"),
    ("juice", "Food"),
    ("food", "Food"),

    # More varied examples
    ("grocery shopping", "Groceries"),
    ("supermarket groceries", "Groceries"),
    ("body lotion", "Personal Care"),
    ("medical prescription", "Healthcare"),
    ("running shoes", "Clothing"),
    ("computer monitor", "Electronics"),
    ("train ticket", "Transportation"),
    ("tuition fees", "Education"),
    ("cleaning supplies", "Household"),
    ("meal", "Food"),
]


training_items = [item for item, category in training_data]
training_categories = [category for item, category in training_data]

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    strip_accents="unicode"
)

X = vectorizer.fit_transform(training_items)

model = LogisticRegression(max_iter=1000)
model.fit(X, training_categories)


def classify_item(item):
    if not item or not str(item).strip():
        return "Unknown"

    name = str(item).strip().lower()

    # Handle invoice adjustments explicitly
    if any(word in name for word in ("discount", "rebate", "coupon")):
        return "Discount / Adjustment"

    # Generic template descriptions do not reveal the actual product
    generic_names = {
        "your products",
        "product",
        "products",
        "item",
        "items",
        "unknown"
    }

    if name in generic_names:
        return "Other / Unclassified"

    item_vector = vectorizer.transform([name])

    # Avoid presenting a weak prediction as a reliable category
    probabilities = model.predict_proba(item_vector)[0]
    best_index = probabilities.argmax()
    confidence = probabilities[best_index]

    if confidence < 0.35:
        return "Other / Unclassified"

    return model.classes_[best_index]
