# Resume Analyzer NLP Service

A pretty small scale NLP-based service that analyzes resumes and provides detailed feedback on their content, structure, and effectiveness. Built with FastAPI and powered by BERT-based deep learning models.

## Features

- Resume analysis for PDF and DOCX formats
- Deep learning-based scoring using BERT model
- Entity extraction for key resume components
- Detailed feedback on different resume sections
- Readability analysis
- Overall scoring system
- RESTful API interface

## Tech Stack

- Python 3.11
- FastAPI
- TensorFlow/Keras
- BERT (bert-base-uncased)
- PDF and DOCX processing libraries
- Docker support

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (optional, for containerized deployment)

## Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/Kingdawnage/nlp-service.git
cd nlp-service
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run locally:
```bash
python3.exe -m app.main
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t resume-analyzer .
```

2. Run using Docker Compose:
```bash
docker-compose up
```

3. (Alt)Run using Docker Compose (with build):
```bash
docker-compose up -build # Skip 1 and 2 if you choose this
```

## Usage

### API Endpoints

1. Health Check:
```
GET /
```

2. Resume Analysis:
```
POST /analyze_resume/
Content-Type: multipart/form-data

file: <resume_file>
```

### Example Response

```json
{
    "entities": {
        "skills": [...],
        "experience": [...],
        "education": [...]
    },
    "scores": {
        "overall": 0.85,
        "model_score": 0.82,
        "readability": 0.88,
        "section_scores": {
            "experience": 0.9,
            "education": 0.85,
            "skills": 0.8
        }
    },
    "feedback": {
        "general": [...],
        "sections": {
            "experience": [...],
            "education": [...],
            "skills": [...]
        }
    },
    "meta": {
        "timestamp": "2024-03-24T12:00:00Z",
        "model_version": "1.0.0"
    }
}
```

## Project Structure

```
nlp-service/
├── app/
│   ├── main.py          # FastAPI application and endpoints
│   ├── model.py         # BERT-based model implementation
│   ├── preprocess.py    # Text processing and analysis utilities
│   └── models/          # Saved model files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Development

The service uses a BERT-based model for resume analysis, which is pre-trained and saved in the `app/models` directory. The model can be retrained if needed by uncommenting the training code in `main.py`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to improve the NLP Resume Analysis Service! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Create a Pull Request

Please make sure to update tests as appropriate and follow the existing code style.