import re
import pdfplumber

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber for better layout handling.
    """
    text_chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            # Extract text with explicit line breaks and layout preservation
            page_text = page.extract_text(
                x_tolerance=3,
                y_tolerance=3,
                layout=True,
                keep_blank_chars=True
            )
            if page_text:
                # Split by newlines and preserve non-empty lines
                lines = [line.strip() for line in page_text.splitlines()]
                text_chunks.extend(lines)
    
    # Join with explicit newlines, removing consecutive empty lines
    return '\n'.join(line for i, line in enumerate(text_chunks) 
                    if line or (i > 0 and i < len(text_chunks)-1 and text_chunks[i-1]))

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a .docx file using python-docx"""
    import docx
    doc = docx.Document(file_path)
    
    text_chunks = []
    for para in doc.paragraphs:
        # Skip empty paragraphs
        if not para.text.strip():
            continue
            
        # Check if text is bold
        is_bold = any(run.bold for run in para.runs)
        
        # Add markdown-style bold if necessary
        text = para.text.strip()
        if is_bold:
            text = f"**{text}**"
            
        text_chunks.append(text)
    
    # Join with explicit newlines, removing consecutive empty lines
    return '\n'.join(text_chunks)

def clean_text(raw_text: str) -> str:
    """Clean the extracted text by removing non-ASCII characters and extra whitespaces"""
    # Remove non-ASCII characters but preserve basic formatting
    text = re.sub(r'[^\x00-\x7F\n]+', ' ', raw_text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove extra spaces while preserving line breaks
    lines = text.split('\n')
    cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
    
    # Remove consecutive empty lines
    result = []
    prev_empty = False
    for line in cleaned_lines:
        if line or not prev_empty:
            result.append(line)
        prev_empty = not line
    
    return '\n'.join(result)

def segment_resume(cleaned_text):
    """
    Segment the resume text into sections
    Assumes sections are separated by headers like Education, Experience, etc.
    """
    sections = re.split(r'(Name|Profile|Education|Experience|Skills|Projects|Certifications)', cleaned_text, flags=re.IGNORECASE)
    segments = {}
    for i in range(1, len(sections), 2):
        header = sections[i].strip().capitalize()
        content = sections[i+1].strip() if i+1 < len(sections) else ''
        segments[header] = content
    return segments

def extract_name(text: str) -> str:
    """Extract name from the resume text using pattern matching."""
    # Common name patterns with improved regex
    patterns = [
        # Bold or formatted name at start (more flexible pattern)
        r'^\s*(?:\*\*|\s)?([A-Z][a-zA-Z\'-]+(?:\s+(?:[A-Z]\.?\s+)?[A-Z][a-zA-Z\'-]+){1,2})(?:\*\*|\s)?\s*$',
        # Pattern for "Name: John Doe" or similar
        r'(?i)name[\s:-]+([A-Z][a-zA-Z\'-]+(?:\s+(?:[A-Z]\.?\s+)?[A-Z][a-zA-Z\'-]+){1,2})',
        # Pattern for names with accents or special characters
        r'^([A-Z][a-zA-ZÀ-ÿ\'-]+(?:\s+(?:[A-Z]\.?\s+)?[A-Z][a-zA-ZÀ-ÿ\'-]+){1,2})\s*$',
        # All caps name pattern
        r'^([A-Z][A-Z\'-]+(?:\s+(?:[A-Z]\.?\s+)?[A-Z][A-Z\'-]+){1,2})\s*$'
    ]
    
    # Split into lines and clean
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Debug output
    print("First 3 lines of resume:")
    for i, line in enumerate(lines[:3]):
        print(f"Line {i + 1}: '{line}'")
    
    # Try each pattern on first few lines
    for line in lines[:3]:
        # Remove any markdown-style formatting and extra spaces
        cleaned_line = re.sub(r'\*\*|__', '', line).strip()
        
        # Debug output
        print(f"\nTrying patterns on: '{cleaned_line}'")
        
        for pattern in patterns:
            match = re.search(pattern, cleaned_line)
            if match:
                name = match.group(1).strip()
                # Convert to proper case if all caps
                if name.isupper():
                    name = ' '.join(word.capitalize() for word in name.split())
                print(f"Found name: {name}")
                return name
    
    # Fallback: check first line with more lenient criteria
    if lines:
        first_line = re.sub(r'\*\*|__', '', lines[0]).strip()
        words = first_line.split()
        if (2 <= len(words) <= 3 and 
            all(word[0].isupper() for word in words) and
            not any(word.lower() in ['resume', 'cv', 'curriculum', 'vitae'] for word in words)):
            name = ' '.join(word.capitalize() for word in words)
            print(f"Found name using fallback: {name}")
            return name
    
    print("No name found, returning Unknown")
    return "Unknown"

def extract_entities(text: str) -> dict:
    """
    Extract entities from the resume text using a simple rule-based approach.
    For demonstration, assume the first line is the candidate's name and use segmentation for other sections.
    """
    entities = {}
    entities["Name"] = extract_name(text)
    
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
    required_sections = ["Name", "Profile", "Education", "Experience", "Skills"]
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