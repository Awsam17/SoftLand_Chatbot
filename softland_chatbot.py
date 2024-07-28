from sentence_transformers import SentenceTransformer, models, losses, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sqlite3
from fuzzywuzzy import fuzz
import pyodbc
from flask import Flask,render_template,request,jsonify
import logging

logging.basicConfig(level=logging.ERROR, filename='app.log', filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s')

word_embedding_model = models.Transformer('C:\\Users\\DELL\\Desktop\\arabert')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

class LearningChatbot :
    def __init__(self, responses_file="chatbot_data.txt"):
        self.model = model
        self.responses_file = responses_file
        self.data = self.read_questions_answers()
        self.question_embeddings = self.model.encode(self.data['questions'])
        self.connection = self.connect_to_database()

    def read_questions_answers(self):
        questions_answers = {'questions': [], 'answers': []}
        with open(self.responses_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(0, len(lines), 2):
                questions_answers['questions'].append(lines[i].strip())
                questions_answers['answers'].append(lines[i + 1].strip())
        return questions_answers

    def save_responses(self):
        with open(self.responses_file, "w", encoding="utf-8") as file:
            for question, answer in zip(self.data['questions'], self.data['answers']):
                file.write(f"{question}\n{answer}\n")

    def connect_to_database(self):
        # Connect to SQL Server
        server_name = 'DESKTOP-GP37IP9'
        database_name = 'SoftLand'
        connection_string = f'DRIVER={{SQL Server}};SERVER={server_name};DATABASE={database_name};Trusted_Connection=yes;'
        return pyodbc.connect(connection_string)
    
    def execute_sql_query(self,question,type):
        columns = ['جلد', 'نعل', 'تيلا', 'توكسون', 'بطانة', 'كتانة', 'اغو', 'بوسطون', 'خيطان', 'شواط']
        arr = ['قدم','طقم','طبق','طبق','قدم','متر','تنكة','كيلو','كيلو','ربطة']
        best_match = None
        best_similarity = -1

        # Iterate over each column and find the best match
        i = 0
        measure_index = 0 
        for column in columns:
            similarity = max(fuzz.token_sort_ratio(word, column) for word in question.split())

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = column
                measure_index = i

            i = i+1

        if best_match is None:
            print("No valid column found in the question.")
            return
        c = self.connection.cursor()
        if (type == 'quantity'):
            # conn = sqlite3.connect('SoftLand.db')
            # c = conn.cursor()
            query = f"SELECT SUM(Quantity) FROM {best_match}"
        else :
            # conn = sqlite3.connect('SoftLand.db')
            # c = conn.cursor()
            query = f"SELECT Color FROM {best_match}"
        try:
            c.execute(query)
            result = c.fetchall()

            if result:
                if (type == 'quantity') :
                    return f"كمية ال{best_match} في الورشة : {result[0][0]} {arr[measure_index]}"
                else :
                    res = f"الوان ال{best_match} المتاحة في الورشة : "
                    colors = [str(color[0]) for color in result]
                    if len(colors) == 0:
                        return f"لا يوجد {best_match} في الورشة."
                    else :
                        res += " و ".join(colors) 
                        return res       
            else:
                return "No data found for {best_match}."
        except pyodbc.ProgrammingError as e:
            # Log the detailed error message
            logging.error(f"Database error occurred: {str(e)}")

            # Return a user-friendly message
            return "عذرًا، حدث خطأ أثناء محاولة استرجاع المعلومات. يرجى المحاولة مرة أخرى لاحقًا."
        except Exception as e:
            # Log any other unexpected exceptions
            logging.error(f"Unexpected error: {str(e)}")

            # Return a user-friendly message for unexpected errors
            return "حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى لاحقًا."

    def classify_question(self, question):
        quantity_keywords = ['كم يوجد', 'كم عدد', 'اديش كمية','ما كمية']
        color_keywords = ['شو لون', 'لون']
        keywords = quantity_keywords + color_keywords

        def generate_ngrams(text, n):
            words = text.split()
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
            return ngrams

        max_quantity_similarity = 0
        max_color_similarity = 0

        for keyword in keywords:
            keyword_length = len(keyword.split())
            question_ngrams = generate_ngrams(question, keyword_length)

            if not question_ngrams:  # Check if question_ngrams is empty
                continue

            keyword_embedding = self.model.encode(keyword)
            ngram_embeddings = self.model.encode(question_ngrams)

            if len(ngram_embeddings) == 0:  # Check if ngram_embeddings is empty
                continue

            cosine_similarities = util.cos_sim(keyword_embedding, ngram_embeddings)
            max_keyword_similarity = max(cosine_similarities[0])

            if keyword in quantity_keywords:
                max_quantity_similarity = max(max_quantity_similarity, max_keyword_similarity)
            elif keyword in color_keywords:
                max_color_similarity = max(max_color_similarity, max_keyword_similarity)

        similarity_threshold = 0.8
        if max_quantity_similarity > similarity_threshold and max_quantity_similarity > max_color_similarity:
            return 'quantity'
        elif max_color_similarity > similarity_threshold:
            return 'color'
        else:
            return 'none'

    def get_response(self,user_input):
        if self.classify_question(user_input) != 'none':
            if self.classify_question(user_input) == 'quantity' :
              return self.execute_sql_query(user_input,'quantity')
            else :
              return self.execute_sql_query(user_input,'color')
        else :
        # Encode user input
            user_embedding = self.model.encode([user_input])

            # Compute cosine similarity between user input and encoded questions
            similarities = cosine_similarity(user_embedding, self.question_embeddings)

            # Find the most similar question
            most_similar_idx = np.argmax(similarities)
            return self.data['answers'][most_similar_idx]

    # def chat(self):
    #     user_input = ""
    #     while user_input.lower() != "exit":
    #         try:
    #             user_input = input("You: ")
    #             # print(self.get_response(user_input))
    #         except KeyboardInterrupt:
    #             print("\nGoodbye!")
    #             self.save_responses()  # Save responses before exiting
    #             exit()
    #     return self.get_response(user_input)

chatbot = LearningChatbot()
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    response = chatbot.get_response(input)
    return response

if __name__ == "__main__":
    app.run(debug=True)

