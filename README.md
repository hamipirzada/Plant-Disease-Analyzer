# Plant Disease Analyzer

A Flask-based web application for analyzing plant diseases using AI. Upload images of plants to detect diseases and get treatment recommendations.

## Features
- Upload and analyze plant images
- AI-powered disease detection
- Treatment recommendations
- Analysis history tracking
- Mobile-friendly interface

## Tech Stack
- Backend: Flask (Python 3.9)
- Frontend: HTML, JavaScript, TailwindCSS
- Image Processing: OpenCV, Pillow

## Setup

1. Clone the repository:
```bash
git clone https://github.com/hamipirzada/Plant-Disease-Analyzer.git
cd Plant-Disease-Analyzer
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create .env file:
```
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key
UPLOAD_FOLDER=uploads
```

5. Run the application:
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

## Project Structure
```
Plant-Disease-Analyzer/
├── app.py              # Main Flask application
├── templates/          # HTML templates
│   └── index.html     # Main template
├── uploads/           # Image upload directory
└── requirements.txt   # Python dependencies
```

## License
MIT License
