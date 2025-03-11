import re
from pdfminer.high_level import extract_text
import pdfplumber

# def extract_text_from_pdf(file_path: str) -> str:
#     """Extract text from a PDF file using pdfminer.six"""
#     return extract_text(file_path)
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber for better layout handling.
    """
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a .docx file using python-docx"""
    import docx
    doc = docx.Document(file_path)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

def clean_text(raw_text: str) -> str:
    """Clean the extracted text by removing non-ASCII characters and extra whitespaces"""
    text = re.sub(r'[^\x00-\x7F]+', ' ', raw_text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def segment_resume(cleaned_text):
    """
    Segment the resume text into sections
    Assumes sections are separated by headers like Education, Experience, etc.
    """
    sections = re.split(r'(Summary|Education|Experience|Skills|Projects|Certifications)', cleaned_text, flags=re.IGNORECASE)
    segments = {}
    for i in range(1, len(sections), 2):
        header = sections[i].strip().capitalize()
        content = sections[i+1].strip() if i+1 < len(sections) else ''
        segments[header] = content
    return segments

def extract_entities(text: str) -> dict:
    """
    Extract entities from the resume text using a simple rule-based approach.
    For demonstration, assume the first line is the candidate's name and use segmentation for other sections.
    """
    # Split text into lines and assume the first non-empty line is the candidate's name.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    entities = {}
    if lines:
        entities["Name"] = lines[0]
    
    # Use segmentation to extract sections.
    sections = segment_resume(text)
    for header, content in sections.items():
        entities[header] = content  # In a real solution, you might further process each section.
    return entities

def generate_feedback(entities: dict, overall_score: float) -> str:
    """
    Generate feedback based on the presence of key sections and the overall resume score.
    This is a simple placeholder logic.
    """
    feedback = []
    required_sections = ["Education", "Experience", "Skills"]
    for section in required_sections:
        if section not in entities or not entities[section]:
            feedback.append(f"Consider adding more detail to your {section.lower()} section.")
    
    # Provide feedback based on the overall score.
    # (Note: With an untrained model, this score is arbitrary; adjust thresholds as needed.)
    if overall_score >= 0:
        feedback.append("Your resume seems well-structured; consider quantifying your achievements.")
    else:
        feedback.append("Consider revising the overall structure for clarity.")
    
    return " ".join(feedback)