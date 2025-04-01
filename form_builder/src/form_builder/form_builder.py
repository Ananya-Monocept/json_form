from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Get Hugging Face token
HF_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')

# Define a fallback model that's more accessible
DEFAULT_MODEL = "microsoft/phi-2"  # Better model for structured outputs

class Control(BaseModel):
    id: str
    type: str
    label: str
    required: bool = False
    validation: Optional[Dict] = None

class Section(BaseModel):
    id: str
    title: str
    controls: List[Control] = []

class Form(BaseModel):
    sections: List[Section] = []

class FormBuilder:
    def __init__(self, model_name: str = None):
        # Use the provided model or default to a more accessible one
        if model_name is None:
            model_name = DEFAULT_MODEL
            print(f"Using default model: {DEFAULT_MODEL}")
        
        try:
            print(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            print("Model loaded successfully")
            
            # Create a text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512
            )
            print("Text generation pipeline created")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Always fallback to GPT-2 as a last resort
            model_name = "gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create a text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512
            )
            print(f"Fallback to {model_name} model")
            
        self.form = Form(sections=[])
        self._control_counter = 0

    def _generate_control_id(self) -> str:
        self._control_counter += 1
        return f"control_{self._control_counter}"

    def add_section(self, title: str) -> Dict:
        section_id = f"section_{len(self.form.sections)}"
        new_section = Section(id=section_id, title=title)
        self.form.sections.append(new_section)
        return {"status": "success", "section_id": section_id}

    def add_control(self, section_id: str, control_type: str, label: str, 
                   required: bool = False, validation: Optional[Dict] = None) -> Dict:
        for section in self.form.sections:
            if section.id == section_id:
                control = Control(
                    id=self._generate_control_id(),
                    type=control_type,
                    label=label,
                    required=required,
                    validation=validation
                )
                section.controls.append(control)
                return {"status": "success", "control_id": control.id}
        return {"error": "Section not found"}

    def update_control(self, control_id: str, **kwargs) -> Dict:
        for section in self.form.sections:
            for control in section.controls:
                if control.id == control_id:
                    for key, value in kwargs.items():
                        setattr(control, key, value)
                    return {"status": "updated"}
        return {"error": "Control not found"}

    def delete_control(self, control_id: str) -> Dict:
        for section in self.form.sections:
            section.controls = [c for c in section.controls if c.id != control_id]
        return {"status": "deleted"}

    def get_form(self) -> Dict:
        return self.form.model_dump()

    def generate_form(self, form_description: str) -> Dict[str, Any]:
        prompt = f"""You are a JSON schema generator. Given a form description, you will output a valid JSON schema for that form.
        
        FORM DESCRIPTION: {form_description}
        
        RULES:
        1. Return ONLY valid JSON with no additional text or explanation
        2. Use the following structure:
           {{
             "type": "object",
             "properties": {{
               "fieldName": {{ "type": "string", "required": true }},
               "anotherField": {{ "type": "number", "minimum": 0 }}
             }}
           }}
        3. Common field types: string, number, boolean
        4. For validation, use: required, format, minimum, maximum, minLength, maxLength
        
        YOUR JSON SCHEMA:
        """
        
        try:
            # Use the pipeline for text generation
            results = self.generator(
                prompt, 
                max_new_tokens=500, 
                do_sample=True, 
                temperature=0.3,  # Lower temperature for more focused outputs
                top_p=0.95,
                num_return_sequences=1
            )
            
            response_text = results[0]['generated_text']
            # Remove the original prompt from the response
            if prompt in response_text:
                response_text = response_text[len(prompt):].strip()
            
            # Extract JSON from response
            try:
                # Try to find JSON in markdown code block
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].strip() 
                else:
                    # Just try to parse the whole thing
                    json_str = response_text.strip()
                
                # Try to clean up any leading/trailing text
                first_brace = json_str.find('{')
                last_brace = json_str.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    json_str = json_str[first_brace:last_brace+1]
                
                json_obj = json.loads(json_str)
                print("Successfully parsed JSON response")
                return json_obj
            except Exception as json_error:
                print(f"Error parsing JSON: {str(json_error)}")
                print(f"Raw response: {response_text}")
                # Return a simple fallback schema if JSON parsing fails
                return {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string", "required": True},
                    }
                }
        except Exception as e:
            print(f"Error generating form: {str(e)}")
            # Return a simple fallback schema
            return {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "required": True},
                }
            }

class FormGenerator:
    def __init__(self):
        self.form_builder = FormBuilder()
    
    def generate_form_schema(self, description: str) -> Dict[str, Any]:
        return self.form_builder.generate_form(description)
        
    def generate_with_jsonformer(self, prompt: str) -> Dict[str, Any]:
        # Since we're not using jsonformer anymore, just delegate to generate_form
        return self.form_builder.generate_form(prompt)
        
    def generate(self, prompt: str, tools: List[str] = None) -> List[Dict]:
        """Generate tool calls based on the prompt"""
        # Generate a form schema using the LLM
        form_json = self.form_builder.generate_form(prompt)
        tool_calls = []
        
        # Extract a title from the prompt for the section
        section_title = "Form Section"
        if "form" in prompt.lower():
            # Extract a potential title from the prompt
            words = prompt.split()
            for i, word in enumerate(words):
                if word.lower() == "form" and i > 0:
                    section_title = f"{words[i-1]} Form"
                    break
        
        # Add a section
        tool_calls.append({
            "name": "add_section",
            "parameters": {
                "title": section_title
            }
        })
        
        # Try to extract fields from the generated form
        if "properties" in form_json:
            for field_name, field_info in form_json.get("properties", {}).items():
                # Determine the control type based on field type
                field_type = field_info.get("type", "text").lower()
                control_type = "text"
                
                if field_type == "number" or field_type == "integer":
                    control_type = "number"
                elif field_type == "boolean":
                    control_type = "checkbox"
                elif field_type == "string":
                    # Check if there's a format specified
                    if field_info.get("format", "").lower() == "email":
                        control_type = "email"
                    elif "password" in field_name.lower():
                        control_type = "password"
                
                # Determine if field is required
                is_required = field_info.get("required", False)
                if is_required is None:
                    is_required = False
                    
                # Create validation rules if present
                validation = {}
                for rule in ["minimum", "maximum", "minLength", "maxLength", "pattern", "format"]:
                    if rule in field_info:
                        validation[rule] = field_info[rule]
                
                if not validation:
                    validation = None
                    
                # Add the control to the section
                tool_calls.append({
                    "name": "add_control",
                    "parameters": {
                        "section_id": "section_0",
                        "control_type": control_type,
                        "label": field_name.replace("_", " ").title(),
                        "required": is_required,
                        "validation": validation
                    }
                })
        
        return tool_calls

def validate_form(form_data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    errors = []
    
    def validate_field(value: Any, field_schema: Dict[str, Any], field_name: str) -> None:
        if "type" not in field_schema:
            errors.append(f"Field {field_name} has no type specified")
            return
            
        field_type = field_schema["type"]
        if field_type == "string":
            if not isinstance(value, str):
                errors.append(f"Field {field_name} must be a string")
        elif field_type == "number":
            if not isinstance(value, (int, float)):
                errors.append(f"Field {field_name} must be a number")
        elif field_type == "boolean":
            if not isinstance(value, bool):
                errors.append(f"Field {field_name} must be a boolean")
        elif field_type == "array":
            if not isinstance(value, list):
                errors.append(f"Field {field_name} must be an array")
            elif "items" in field_schema:
                for i, item in enumerate(value):
                    validate_field(item, field_schema["items"], f"{field_name}[{i}]")
    
    for field_name, field_schema in schema.get("properties", {}).items():
        if field_name not in form_data:
            if field_schema.get("required", False):
                errors.append(f"Required field {field_name} is missing")
            continue
            
        validate_field(form_data[field_name], field_schema, field_name)
    
    return errors 