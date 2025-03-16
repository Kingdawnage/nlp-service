from fastapi import FastAPI, File, UploadFile
import uvicorn
import tempfile
import keras
import tf_keras as tkf
from transformers import BertTokenizer, TFBertModel
from app.model import ResumeAnalyzerModel
# from model_test import ResumerAnalyzerModel

from app.model import get_model
from app.preprocess import (
    extract_text_from_pdf,
    extract_text_from_docx,
    clean_text,
    extract_entities,
    generate_feedback
)

app = FastAPI()

## Uncomment to train a new model and save it
model, tokenizer = get_model()
model.save("app/models/model_1.0.0.keras")

# custom_objects = {
#     "ResumeAnalyzerModel": ResumeAnalyzerModel,
# }
# model = keras.models.load_model("app/models/model_1.0.0.keras", custom_objects=custom_objects)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Resume Analyzer API!"}

@app.post("/analyze_resume/")
async def analyze_resume(file: UploadFile = File(...)):
    # Save the uploaded file temporarily.
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        file_path = tmp.name
    
    # Determine file type by its extension and extract raw text.
    if file.filename.endswith('.pdf'):
        raw_text = extract_text_from_pdf(file_path)
    elif file.filename.endswith('.docx'):
        raw_text = extract_text_from_docx(file_path)
    else:
        # Assume plain text for other file types.
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    
    # Clean the extracted text.
    cleaned = clean_text(raw_text)
    
    # Tokenize the cleaned text for model input.
    encoded_input = dict(tokenizer(cleaned, return_tensors='tf', padding=True, truncation=True))
    
    # Run the model inference.
    token_logits, overall_score = model(encoded_input)
    overall_score_val = overall_score.numpy().tolist()[0][0]
    
    # Extract entities using our rule-based approach.
    entities = extract_entities(cleaned)
    
    # Generate feedback based on the extracted entities and overall score.
    feedback_text = generate_feedback(entities, overall_score_val)
    
    # Construct the JSON output.
    output_json = {
        "entities": entities,
        "scores": {"overall": overall_score_val},
        "feedback": feedback_text,
        "debug": {"token_logits_shape": token_logits.shape.as_list()}
    }
    
    return output_json

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)