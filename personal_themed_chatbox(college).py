# ============================================================
#              COLLEGE ENQUIRY CHATBOT
#        Complete Machine Learning + NLP Project
#                  Single Python File
# ============================================================

# Install required libraries:
# pip install pandas scikit-learn

import random
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. EMBEDDED DATASET
# 25 INTENTS - 520 QUESTION ANSWER TRAINING PAIRS
# ============================================================

random.seed(42)

# Each intent has 4 question topics.
# Each topic has 5 natural-language variations.
# 4 x 5 = 20 examples per intent.

patterns = [
    "Can you tell me about {}?",
    "I need information about {}.",
    "Please explain {}.",
    "Where can I find details about {}?",
    "Could you help me with {}?"
]

college_data = {

    "college_info": (
        "Our college provides quality education, academic "
        "programs, research opportunities and student "
        "development activities.",
        [
            "the college",
            "the institution",
            "campus facilities",
            "the college overview"
        ]
    ),

    "admission": (
        "The admission process generally involves submitting "
        "an application, checking eligibility, document "
        "verification and fee payment. Contact the admission "
        "office for current procedures.",
        [
            "the admission process",
            "applying to this college",
            "new student admission",
            "college enrolment"
        ]
    ),

    "eligibility": (
        "Eligibility depends on the selected course and "
        "applicable admission rules. Contact the admission "
        "office for exact requirements.",
        [
            "admission eligibility",
            "minimum qualifying marks",
            "course admission requirements",
            "educational qualifications"
        ]
    ),

    "courses": (
        "The college offers various undergraduate and "
        "postgraduate programs. Contact the academic office "
        "for the current course list.",
        [
            "available courses",
            "undergraduate programs",
            "postgraduate programs",
            "degree programs"
        ]
    ),

    "departments": (
        "Department information, faculty details and "
        "academic programs are available through the "
        "respective department offices.",
        [
            "college departments",
            "academic departments",
            "department information",
            "department facilities"
        ]
    ),

    "fees": (
        "College fees depend on the course, academic year "
        "and admission category. Contact the accounts office "
        "for the official fee structure.",
        [
            "college fees",
            "tuition fees",
            "semester fee structure",
            "annual course fees"
        ]
    ),

    "fee_payment": (
        "Fee payment methods and deadlines should be "
        "confirmed with the accounts office. Ask them "
        "about online payments and instalments.",
        [
            "online fee payment",
            "paying college fees",
            "semester fee payment",
            "fee payment methods"
        ]
    ),

    "scholarship": (
        "Eligible students may apply for government, "
        "merit-based and institutional scholarships. "
        "Contact the scholarship office for details.",
        [
            "college scholarships",
            "government scholarships",
            "merit scholarships",
            "financial assistance"
        ]
    ),

    "hostel": (
        "Hostel facilities may be available for students. "
        "Contact the hostel office for accommodation, "
        "room availability, fees and rules.",
        [
            "hostel facilities",
            "hostel accommodation",
            "hostel room availability",
            "hostel fees and rules"
        ]
    ),

    "transport": (
        "College transportation services and bus routes "
        "can be confirmed through the transport office.",
        [
            "college bus facilities",
            "college bus routes",
            "student transportation",
            "college bus timings"
        ]
    ),

    "library": (
        "The college library provides academic books, "
        "journals and learning resources. Contact library "
        "staff for membership and timings.",
        [
            "library facilities",
            "library membership",
            "library books",
            "library working hours"
        ]
    ),

    "placement": (
        "The placement cell supports students through "
        "career guidance, interview preparation, "
        "recruitment activities and training.",
        [
            "campus placements",
            "placement training",
            "placement opportunities",
            "recruitment companies"
        ]
    ),

    "internship": (
        "Internship opportunities may be available through "
        "academic departments and industry connections. "
        "Contact your department or placement cell.",
        [
            "student internships",
            "summer internships",
            "internship opportunities",
            "internship application process"
        ]
    ),

    "examination": (
        "Examination schedules, registration and rules "
        "are announced by the examination department.",
        [
            "semester examinations",
            "examination timetable",
            "exam registration",
            "examination rules"
        ]
    ),

    "results": (
        "Examination results are generally published "
        "through the official student portal or examination "
        "department.",
        [
            "semester examination results",
            "checking exam results",
            "student marksheets",
            "result publication"
        ]
    ),

    "attendance": (
        "Attendance requirements depend on institutional "
        "and university regulations. Contact your department "
        "for your official attendance status.",
        [
            "student attendance",
            "minimum attendance requirements",
            "attendance percentage",
            "attendance shortage"
        ]
    ),

    "faculty": (
        "Faculty information and contact details can be "
        "obtained from the respective academic departments.",
        [
            "college faculty",
            "teaching staff",
            "head of department",
            "faculty contact information"
        ]
    ),

    "timetable": (
        "Class timetables are generally shared by academic "
        "departments or through the student portal.",
        [
            "class timetable",
            "weekly class schedule",
            "daily lecture timings",
            "laboratory timetable"
        ]
    ),

    "events": (
        "The college organizes academic activities, "
        "workshops, seminars and cultural events. "
        "Check official notices for upcoming events.",
        [
            "college events",
            "cultural programs",
            "technical symposiums",
            "college workshops"
        ]
    ),

    "sports": (
        "Sports facilities and student activities depend "
        "on campus resources. Contact the physical "
        "education department for details.",
        [
            "college sports facilities",
            "sports competitions",
            "indoor games",
            "outdoor sports activities"
        ]
    ),

    "canteen": (
        "Canteen services and food availability can be "
        "confirmed with the college administration.",
        [
            "college canteen",
            "campus food facilities",
            "canteen food menu",
            "canteen working hours"
        ]
    ),

    "wifi": (
        "Campus internet and Wi-Fi availability depend "
        "on institutional facilities. Contact IT support "
        "for connection assistance.",
        [
            "campus wifi",
            "college internet facilities",
            "student wifi access",
            "internet connection support"
        ]
    ),

    "documents": (
        "Students can contact the administration office "
        "to request bonafide certificates, transfer "
        "certificates, transcripts and other documents.",
        [
            "bonafide certificate",
            "transfer certificate",
            "study certificate",
            "official college documents"
        ]
    ),

    "contact": (
        "For official enquiries, please refer to the "
        "college website or contact the appropriate "
        "administrative office.",
        [
            "college contact information",
            "college phone number",
            "college email address",
            "administrative office contact"
        ]
    )
}


# ============================================================
# 2. GENERATE THE EMBEDDED TRAINING EXAMPLES
# ============================================================

dataset = []

for intent, (answer, topics) in college_data.items():

    for topic in topics:

        for pattern in patterns:

            question = pattern.format(topic)

            dataset.append({
                "intent": intent,
                "question": question,
                "answer": answer
            })


# Greeting examples (20)

greeting_questions = [
    "hello", "hi", "hey", "good morning",
    "good afternoon", "good evening",
    "hello there", "hi there", "how are you",
    "can you help me", "start chat",
    "I need some information", "greetings",
    "is anyone available", "help me please",
    "I have a question", "hey chatbot",
    "good day", "can we talk",
    "I want to ask something"
]

for question in greeting_questions:
    dataset.append({
        "intent": "greeting",
        "question": question,
        "answer": "Hello! Welcome to our college chatbot. "
                  "How can I help you today?"
    })


# Fallback examples (20)

fallback_questions = [
    "what is the meaning of life",
    "tell me a joke",
    "what is the weather today",
    "who are you",
    "what can you do",
    "I want to talk to a person",
    "tell me something interesting",
    "I have a different question",
    "please help me with something else",
    "that is not what I asked",
    "I do not understand",
    "can you explain something else",
    "I need human assistance",
    "say something",
    "can you repeat that",
    "this is confusing",
    "I have another doubt",
    "please clarify",
    "what is your purpose",
    "I cannot find my answer"
]

for question in fallback_questions:
    dataset.append({
        "intent": "fallback",
        "question": question,
        "answer": "Sorry! I couldn't understand your question. "
                  "Please ask a college-related question or "
                  "contact the college office."
    })


# Convert embedded dataset to DataFrame

df = pd.DataFrame(dataset)

assert len(df) == 520, "Dataset must contain exactly 520 rows!"

print("\n================================")
print("COLLEGE CHATBOT DATASET")
print("================================")

print("Total questions:", len(df))
print("Total intents:", df["intent"].nunique())

print("\nIntent distribution:")
print(df["intent"].value_counts())


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["question"] = df["question"].apply(preprocess_text)


# ============================================================
# 4. MACHINE LEARNING MODEL
# ============================================================

X = df["question"]
y = df["intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


print("\nTraining the chatbot model...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 5. MODEL EVALUATION
# ============================================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print("MODEL EVALUATION")
print("================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ============================================================
# 6. CHATBOT RESPONSE FUNCTION
# ============================================================

def chatbot_response(user_input):

    if not user_input.strip():
        return "Please enter a question."

    cleaned_input = preprocess_text(user_input)

    # Handle common greetings directly
    if cleaned_input in [
        "hello", "hi", "hey", "good morning",
        "good afternoon", "good evening"
    ]:
        return (
            "Hello! Welcome to RG college chatbot. "
            "How can I help you today?"
        )

    # Predict the user's intent
    probabilities = model.predict_proba([cleaned_input])[0]

    confidence = max(probabilities)

    predicted_intent = model.classes_[
        probabilities.argmax()
    ]

    # Confidence threshold for unknown questions
    if confidence < 0.30:
        return (
            "Sorry! I couldn't understand your question. "
            "Please ask a college-related question."
        )

    # Return the matching answer
    if predicted_intent == "greeting":
        return (
            "Hello! Welcome to our college chatbot. "
            "How can I help you today?"
        )

    if predicted_intent == "fallback":
        return (
            "Sorry! I can mainly answer college-related "
            "questions. Please ask about admissions, "
            "courses, fees, placements or facilities."
        )

    answer = college_data[0]

    return answer


# ============================================================
# 7. INTERACTIVE CHATBOT
# ============================================================

def start_chatbot():

    print("\n")
    print("========================================")
    print("       COLLEGE ENQUIRY CHATBOT")
    print("========================================")

    print("Welcome! Ask me anything about college.")
    print("Type 'exit' to close the chatbot.")
    print("----------------------------------------")

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print(
                "\nBot: Thank you for using the "
                "College Enquiry Chatbot!"
            )
            break

        response = chatbot_response(user_input)

        print("\nBot:", response)


# ============================================================
# 8. RUN THE PROJECT
# ============================================================

if __name__ == "__main__":

    start_chatbot()