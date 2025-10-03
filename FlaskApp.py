from flask import Flask, request, jsonify
from pydantic import BaseModel
from typing import List, Dict
from deep_translator import GoogleTranslator
import uuid
import json
import os
from flask_cors import CORS
from dotenv import load_dotenv

# Google Gemini SDK for LLM integration
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Gemini API
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=gemini_api_key)

# Initialize Gemini model
# Using gemini-1.5-flash for fast document generation
model = genai.GenerativeModel('gemini-1.5-flash')

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
    """Translate text to Hindi using Google Translator"""
    try:
        translated = GoogleTranslator(source='auto', target='hi').translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return the original text in case of an error

def generate_with_gemini(prompt: str) -> str:
    """Generate content using Gemini API"""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        raise e

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Legal Document Generator API is running"
    }), 200

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    """Generate questions based on document type"""
    try:
        data = request.json
        doc_type = data.get('doc_type')
        language = data.get('language', 'English')

        if not doc_type:
            return jsonify({"error": "doc_type is required"}), 400

        # Create the prompt
        if language == "Hindi":
            question_prompt = translate_to_hindi(
                f"Generate a list of specific questions that will help clarify the names and details needed to create a {doc_type}. "
                f"Focus on gathering names, dates, and other essential information. "
                f"Provide exactly 8-10 clear questions, one per line, numbered."
            )
        else:
            question_prompt = (
                f"Generate a list of specific questions that will help clarify the names and details needed to create a {doc_type}. "
                f"Focus on gathering names, dates, and other essential information. "
                f"Provide exactly 8-10 clear questions, one per line, numbered."
            )

        # Generate questions using Gemini
        questions_text = generate_with_gemini(question_prompt)
        
        # Split and clean questions
        questions_list = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
        
        return jsonify({"questions": questions_list})
    
    except Exception as e:
        return jsonify({
            "error": f"Error generating questions: {str(e)}"
        }), 500

@app.route('/generate_document', methods=['POST'])
def generate_document():
    """Generate a legal document based on answers"""
    try:
        data = request.json
        doc_type = data.get('doc_type')
        doc_title = data.get('title', f"New {doc_type}")
        answers = data.get('answers', {})
        language = data.get('language', 'English')

        if not doc_type:
            return jsonify({"error": "doc_type is required"}), 400

        # First, generate questions to understand the context
        if language == "Hindi":
            question_prompt = translate_to_hindi(
                f"Generate a list of specific questions that will help clarify the names and details needed to create a {doc_type}. "
                f"Focus on gathering names, dates, and other essential information. "
                f"Provide exactly 8-10 clear questions, one per line, numbered."
            )
        else:
            question_prompt = (
                f"Generate a list of specific questions that will help clarify the names and details needed to create a {doc_type}. "
                f"Focus on gathering names, dates, and other essential information. "
                f"Provide exactly 8-10 clear questions, one per line, numbered."
            )

        questions_text = generate_with_gemini(question_prompt)
        questions_list = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]
        
        # Create answers text from provided answers
        answers_text = ' '.join([f"{q}: {answers.get(q, '')}" for q in answers.keys() if answers.get(q)])
        
        # If no answers provided, use all questions with empty values
        if not answers_text:
            answers_text = "No specific details provided. Generate a template document."

        # Generate the document
        if language == "Hindi":
            doc_prompt = translate_to_hindi(
                f"You are a professional legal document expert. Generate a complete and properly formatted {doc_type} "
                f"using the following details:\n\n{answers_text}\n\n"
                f"The document should be:\n"
                f"- Professionally formatted\n"
                f"- Legally sound\n"
                f"- Complete with all necessary sections\n"
                f"- Ready to use\n\n"
                f"Generate the complete document now:"
            )
        else:
            doc_prompt = (
                f"You are a professional legal document expert. Generate a complete and properly formatted {doc_type} "
                f"using the following details:\n\n{answers_text}\n\n"
                f"The document should be:\n"
                f"- Professionally formatted\n"
                f"- Legally sound\n"
                f"- Complete with all necessary sections\n"
                f"- Ready to use\n\n"
                f"Generate the complete document now:"
            )

        document_text = generate_with_gemini(doc_prompt)

        # Create a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Save the document
        document = {
            "id": doc_id,
            "title": doc_title,
            "document_text": document_text,
            "doc_type": doc_type,
            "language": language,
            "answers": answers
        }
        documents[doc_id] = document
        
        return jsonify({"document": document})
    
    except Exception as e:
        return jsonify({
            "error": f"Error generating document: {str(e)}"
        }), 500

@app.route('/documents', methods=['GET'])
def get_documents():
    """Get all documents"""
    return jsonify({"documents": list(documents.values())})

@app.route('/documents/<document_id>', methods=['GET'])
def get_document(document_id):
    """Get a specific document by ID"""
    document = documents.get(document_id)
    if document:
        return jsonify({"document": document})
    return jsonify({"error": "Document not found"}), 404

@app.route('/documents/<document_id>', methods=['PUT'])
def edit_document(document_id):
    """Edit an existing document"""
    try:
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
            
            # Create answers text
            answers_text = ' '.join([f"{q}: {answers.get(q, '')}" for q in answers.keys() if answers.get(q)])
            
            # Generate document prompt
            if language == "Hindi":
                doc_prompt = translate_to_hindi(
                    f"You are a professional legal document expert. Generate a complete and properly formatted {doc_type} "
                    f"using the following details:\n\n{answers_text}\n\n"
                    f"The document should be:\n"
                    f"- Professionally formatted\n"
                    f"- Legally sound\n"
                    f"- Complete with all necessary sections\n"
                    f"- Ready to use\n\n"
                    f"Generate the complete document now:"
                )
            else:
                doc_prompt = (
                    f"You are a professional legal document expert. Generate a complete and properly formatted {doc_type} "
                    f"using the following details:\n\n{answers_text}\n\n"
                    f"The document should be:\n"
                    f"- Professionally formatted\n"
                    f"- Legally sound\n"
                    f"- Complete with all necessary sections\n"
                    f"- Ready to use\n\n"
                    f"Generate the complete document now:"
                )

            document_text = generate_with_gemini(doc_prompt)
            document['document_text'] = document_text
        
        documents[document_id] = document
        return jsonify({"document": document})
    
    except Exception as e:
        return jsonify({
            "error": f"Error updating document: {str(e)}"
        }), 500

@app.route('/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    """Delete a document"""
    if document_id not in documents:
        return jsonify({"error": "Document not found"}), 404
    
    del documents[document_id]
    return jsonify({"success": True, "message": "Document deleted successfully"})

@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate text to Hindi"""
    try:
        data = request.json
        text = data.get('text')
        target_language = data.get('target', 'hi')
        
        if not text:
            return jsonify({"error": "text is required"}), 400
        
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return jsonify({"translated_text": translated})
    
    except Exception as e:
        return jsonify({
            "error": f"Translation error: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred on the server"
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)