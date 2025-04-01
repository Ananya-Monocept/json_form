# Form Builder API

A FastAPI application that generates form schemas based on text descriptions using language models.

## Features

- Generate form schemas from text descriptions
- Support for different input methods (direct and tool-based)
- Form validation
- RESTful API for form manipulation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/form-builder.git
cd form-builder
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r src/requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the `src` directory with:
```
HUGGINGFACE_HUB_TOKEN=your_huggingface_token
```

## Usage

1. Run the API:
```bash
cd src
python api.py
```

2. Access the API documentation at http://localhost:8000/docs

3. Example API call:
```bash
curl -X 'POST' \
  'http://localhost:8000/generate-form' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Create a contact form with name, email, phone number and a message field",
  "method": "tools"
}'
```

## API Endpoints

- `POST /generate-form`: Generate a form based on a description
- `POST /add-section`: Add a section to a form
- `POST /add-control`: Add a control to a section
- `PUT /update-control/{control_id}`: Update a control
- `DELETE /delete-control/{control_id}`: Delete a control
- `GET /get-form`: Get the current form 