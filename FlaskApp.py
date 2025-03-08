from flask import Flask, request, jsonify
from together import Together
from pydantic import BaseModel
from typing import List, Dict
from deep_translator import GoogleTranslator
import uuid
import json
import os
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Together AI
together = Together()

# Store documents in memory (in production, use a database)
documents = {}

class Question(BaseModel):
    question: str
    answer: str

class LegalDocument(BaseModel):
    id: str
    title: str
    document_text: str
    doc_type: str
    language: str
    answers: Dict[str, str]

def translate_to_hindi(text: str) -> str:
    try:
        translated = GoogleTranslator(source='auto', target='hi').translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return the original text in case of an error

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    data = request.json
    doc_type = data.get('doc_type')
    language = data.get('language', 'English')

    if language == "Hindi":
        question_prompt = translate_to_hindi(f"Generate a list of specific questions that will help clarify the names and details needed to create a {doc_type}. Focus on gathering names, dates, and other essential information.")
    else:
        question_prompt = f"Generate a list of specific questions that will help clarify the names and details needed to create a {doc_type}. Focus on gathering names, dates, and other essential information."

    questions_response = together.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
        messages=[{"role": "user", "content": question_prompt}]
    )
    
    questions_list = questions_response.choices[0].message.content.strip().split('\n')
    return jsonify({"questions": questions_list})

@app.route('/generate_document', methods=['POST'])
def generate_document():
    data = request.json
    doc_type = data.get('doc_type')
    doc_title = data.get('title', f"New {doc_type}")
    answers = data.get('answers', {})
    language = data.get('language', 'English')

    # Generate the question prompt based on the document type
    if language == "Hindi":
        question_prompt = translate_to_hindi(f"Generate a list of specific questions that will help clarify the names and details needed to create a {doc_type}. Focus on gathering names, dates, and other essential information.")
    else:
        question_prompt = f"Generate a list of specific questions that will help clarify the names and details needed to create a {doc_type}. Focus on gathering names, dates, and other essential information."

    questions_response = together.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
        messages=[{"role": "user", "content": question_prompt}]
    )
    
    questions_list = questions_response.choices[0].message.content.strip().split('\n')
    answers_text = ' '.join([answers.get(q, '') for q in questions_list])

    if language == "Hindi":
        doc_prompt = translate_to_hindi(f"Generate a {doc_type} using these details: {answers_text}")
    else:
        doc_prompt = f"Generate a {doc_type} using these details: {answers_text}"

    document_response = together.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
        messages=[{"role": "user", "content": doc_prompt}]
    )

    # Create a unique ID for the document
    doc_id = str(uuid.uuid4())
    
    # Save the document
    document = {
        "id": doc_id,
        "title": doc_title,
        "document_text": document_response.choices[0].message.content.strip(),
        "doc_type": doc_type,
        "language": language,
        "answers": answers
    }
    documents[doc_id] = document
    
    return jsonify({"document": document})

@app.route('/documents', methods=['GET'])
def get_documents():
    return jsonify({"documents": list(documents.values())})

@app.route('/documents/<document_id>', methods=['GET'])
def get_document(document_id):
    document = documents.get(document_id)
    if document:
        return jsonify({"document": document})
    return jsonify({"error": "Document not found"}), 404

@app.route('/documents/<document_id>', methods=['PUT'])
def edit_document(document_id):
    if document_id not in documents:
        return jsonify({"error": "Document not found"}), 404
    
    data = request.json
    document = documents[document_id]
    
    # Update document fields
    if 'document_text' in data:
        document['document_text'] = data['document_text']
    if 'title' in data:
        document['title'] = data['title']
    
    # If answers were updated, regenerate the document
    if 'answers' in data and data.get('regenerate', False):
        document['answers'] = data['answers']
        
        language = document['language']
        doc_type = document['doc_type']
        answers = document['answers']
        
        answers_text = ' '.join([answers.get(q, '') for q in answers.keys()])
        
        if language == "Hindi":
            doc_prompt = translate_to_hindi(f"Generate a {doc_type} using these details: {answers_text}")
        else:
            doc_prompt = f"Generate a {doc_type} using these details: {answers_text}"

        document_response = together.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K",
            messages=[{"role": "user", "content": doc_prompt}]
        )
        
        document['document_text'] = document_response.choices[0].message.content.strip()
    
    documents[document_id] = document
    return jsonify({"document": document})

@app.route('/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    if document_id not in documents:
        return jsonify({"error": "Document not found"}), 404
    
    del documents[document_id]
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)