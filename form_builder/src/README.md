# Form Builder with JSONFormer and Tool Calling

A powerful form builder that combines JSONFormer for structured generation with tool-based form manipulation. This project implements a complete end-to-end workflow for generating and manipulating forms using both direct JSON generation and tool-based approaches.

## Features

- **Hybrid Generation**: Use both JSONFormer for direct form generation and tool-based manipulation
- **Structured Output**: Guaranteed valid JSON output through JSONFormer
- **Tool-Based Manipulation**: Precise control over form modifications
- **Validation**: Built-in validation for form structure and field types
- **RESTful API**: FastAPI-based endpoints for all operations
- **Model Fine-tuning**: Support for fine-tuning with Unsloth

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd form-builder
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
form_builder/
├── api.py              # FastAPI application
├── form_builder.py     # Core form builder implementation
├── train.py           # Model training script
├── seed_data.jsonl    # Training data
└── requirements.txt   # Project dependencies
```

## Usage

### 1. Training the Model

To fine-tune the model on your own data:

```bash
python train.py
```

The model will be saved in the `form_builder_model` directory.

### 2. Running the API

Start the FastAPI server:

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### 3. API Endpoints

#### Generate Form
```http
POST /generate-form
Content-Type: application/json

{
    "prompt": "Create a login form with email and password",
    "method": "tools"  // or "direct"
}
```

#### Add Section
```http
POST /add-section?title=Contact Information
```

#### Add Control
```http
POST /add-control
Content-Type: application/json

{
    "section_id": "section_0",
    "control_type": "email",
    "label": "Email",
    "required": true,
    "validation": {
        "type": "regex",
        "pattern": "^\\S+@\\S+\\.\\S+$"
    }
}
```

#### Update Control
```http
PUT /update-control/{control_id}
Content-Type: application/json

{
    "required": false
}
```

#### Delete Control
```http
DELETE /delete-control/{control_id}
```

#### Get Form
```http
GET /get-form
```

## Example Usage

```python
import requests

# Initialize form
response = requests.post("http://localhost:8000/generate-form", 
    json={
        "prompt": "Create a login form with email and password",
        "method": "tools"
    }
)
form = response.json()

# Add a new section
response = requests.post("http://localhost:8000/add-section?title=Additional Info")
section_id = response.json()["section_id"]

# Add a control to the new section
response = requests.post("http://localhost:8000/add-control",
    json={
        "section_id": section_id,
        "control_type": "text",
        "label": "Full Name",
        "required": True
    }
)
```

## Form Structure

The generated forms follow this structure:

```json
{
  "sections": [
    {
      "id": "section_0",
      "title": "Login",
      "controls": [
        {
          "id": "control_1",
          "type": "email",
          "label": "Email",
          "required": true,
          "validation": {
            "type": "regex",
            "pattern": "^\\S+@\\S+\\.\\S+$"
          }
        }
      ]
    }
  ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 