from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, TypedDict, Union
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re
import os
from pathlib import Path
from langgraph.graph import START, END, Graph
from form_models import IForm, IFormSections, IFormControl, IValidator
from langchain_core.runnables.config import RunnableConfig
from fastapi.middleware.cors import CORSMiddleware
import traceback
from copy import deepcopy
# Load environment variables
load_dotenv()

app = FastAPI(title="Form Builder API")
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel(os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'))

class TemplateRequestModel(BaseModel):
    prompt: str
    template_name: str




def clean_null_values(data):
        """Recursively remove all null values from dictionaries and lists."""
        if isinstance(data, dict):
            return {
                key: clean_null_values(value)
                for key, value in data.items()
                if value is not None and clean_null_values(value) not in (None, {}, [])
            }
        elif isinstance(data, list):
            return [
                clean_null_values(item)
                for item in data
                if item is not None and clean_null_values(item) not in (None, {}, [])
            ]
        return data


def transform_json_for_pydantic(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform the JSON structure to match the Pydantic models without modifying the models.
    This function now supports multiple JSON structures:
    1. Existing structure (`formSections` directly under root).
    2. New structure nested under `data → jsonFormData`.
    3. Structure with `form → sections → fields`.
    4. Directly provided `formSections` at the root level.
    """
    # Make a deep copy to avoid modifying the original
    transformed_data = {}

    try:
        # Extract the nested jsonFormData or the new "form" structure
        form_data = (
            json_data.get('data', {}).get('data', {}).get('jsonFormData', {})
            or json_data.get("data", {}).get("jsonFormData", {})
            or json_data.get("form", {})
            or {}
        )

        # If no valid structure is found, check for direct `formSections` at the root level
        if not form_data and "formSections" in json_data:
            form_data = {"formSections": json_data["formSections"]}

        # If still no valid structure is found, raise an error
        if not form_data:
            raise ValueError(f"Could not find valid form data. Available top-level keys: {list(json_data.keys())}")

        # Copy the form_data to our transformed data
        transformed_data = deepcopy(form_data)

        # Handle the new "sections" structure (if present)
        if "sections" in transformed_data:
            # Map "sections" to "formSections"
            transformed_data["formSections"] = []
            for section in transformed_data.pop("sections", []):
                transformed_section = {
                    "sectionTitle": section.get("sectionName", "Untitled Section"),
                    "sectionId": section.get("sectionId", None),
                    "visibleLabel": True,
                    "visible": True,
                    "class_": None,
                    "formControls": []
                }

                seen_names = set()  # To handle duplicate names

                # Map "fields" to "formControls"
                for field in section.get("fields", []):
                    # Skip invisible or disabled controls
                    if field.get("visible", True) is False or field.get("disabled", False) is True:
                        continue

                    # Normalize the name field
                    label = field.get("label", field.get("fieldName", "Unnamed Control"))
                    name = field.get("fieldId", "")
                    if not name:
                        name = to_camel_case(label)

                    # Handle duplicate names
                    if name in seen_names:
                        name = f"{name}_duplicate"
                    seen_names.add(name)

                    transformed_field = {
                        "name": name,
                        "label": label,
                        "visibleLabel": True,
                        "type_": field.get("type", "text"),
                        "validators": [],
                        "visibilityRules": field.get("visibilityRules", None)
                    }

                    # Process validators
                    for validator in field.get("validators", []):
                        transformed_validator = {}
                        if validator.get("type") == "required":
                            transformed_validator["validatorName"] = "required"
                            transformed_validator["required"] = True
                            transformed_validator["message"] = validator.get("message", "This field is required.")
                        elif validator.get("type") == "pattern":
                            transformed_validator["validatorName"] = "pattern"
                            transformed_validator["pattern"] = validator.get("value", "")
                            transformed_validator["message"] = validator.get("message", "Invalid format.")
                        elif validator.get("type") == "minlength":
                            transformed_validator["validatorName"] = "minLength"
                            transformed_validator["minLength"] = validator.get("value", 0)
                            transformed_validator["message"] = validator.get("message", "Input too short.")
                        elif validator.get("type") == "maxlength":
                            transformed_validator["validatorName"] = "maxLength"
                            transformed_validator["maxLength"] = validator.get("value", 0)
                            transformed_validator["message"] = validator.get("message", "Input too long.")
                        if transformed_validator:
                            transformed_field["validators"].append(transformed_validator)

                    # Add options if present
                    if "options" in field:
                        transformed_field["options"] = [
                            {"value": option.get("value"), "label": option.get("label")}
                            for option in field["options"]
                        ]

                    transformed_section["formControls"].append(transformed_field)

                transformed_data["formSections"].append(transformed_section)

        # Handle the existing "formSections" structure
        if "formSections" in transformed_data:
            for section in transformed_data["formSections"]:
                # Rename 'class' to 'class_' if needed
                if "class" in section and "class_" not in section:
                    section["class_"] = section.pop("class")

                seen_names = set()  # To handle duplicate names

                # Process controls in this section
                for control in section.get("formControls", []):
                    # Skip invisible or disabled controls
                    if control.get("visible", True) is False or control.get("disabled", False) is True:
                        continue

                    # Normalize the name field
                    label = control.get("label", "Unnamed Control")
                    name = control.get("name", "")
                    if not name:
                        name = to_camel_case(label)

                    # Handle duplicate names
                    if name in seen_names:
                        name = f"{name}_duplicate"
                    seen_names.add(name)

                    control["name"] = name

                    if "class" in control and "class_" not in control:
                        control["class_"] = control.pop("class")

                    # Standardize validators
                    for validator in control.get("validators", []):
                        if "minlength" in validator:
                            validator["minLength"] = validator.pop("minlength")
                        if "maxlength" in validator:
                            validator["maxLength"] = validator.pop("maxlength")

        # Apply additional transformations
        _transform_class_fields(transformed_data)
        _transform_type_fields(transformed_data)
        _transform_validators(transformed_data)
        _transform_options(transformed_data)

        return transformed_data
    except Exception as e:
        raise ValueError(f"Error transforming JSON: {str(e)}")
# Helper Functions
def to_camel_case(label: str) -> str:
    """Convert a label to camelCase."""
    words = label.strip().lower().split()
    if not words:
        return ""
    return words[0] + "".join(word.capitalize() for word in words[1:])


def _transform_class_fields(data: Dict[str, Any]) -> None:
    """Transform 'class' fields to 'class_' for Pydantic compatibility."""
    if isinstance(data, dict):
        if 'class' in data:
            data['class_'] = data.pop('class')
        for key, value in list(data.items()):
            if isinstance(value, (dict, list)):
                _transform_class_fields(value)
    if isinstance(data, list):  # Changed from elif to if
        for item in data:
            if isinstance(item, (dict, list)):
                _transform_class_fields(item)


def _transform_type_fields(data: Dict[str, Any]) -> None:
    """Transform 'type' fields to 'type_' for Pydantic compatibility."""
    if isinstance(data, dict):
        if 'type' in data:
            data['type_'] = data.pop('type')
        for key, value in list(data.items()):
            if isinstance(value, (dict, list)):
                _transform_type_fields(value)
    if isinstance(data, list):  # Changed from elif to if
        for item in data:
            if isinstance(item, (dict, list)):
                _transform_type_fields(item)


def _transform_validators(data: Dict[str, Any]) -> None:
    """Transform validator fields to match Pydantic model expectations."""
    if isinstance(data, dict):
        if 'validators' in data and isinstance(data['validators'], list):
            for validator in data['validators']:
                if 'minlength' in validator:
                    validator['minLength'] = validator.pop('minlength')
                if 'maxlength' in validator:
                    validator['maxLength'] = validator.pop('maxlength')
        for key, value in list(data.items()):
            if isinstance(value, (dict, list)):
                _transform_validators(value)
    if isinstance(data, list):  # Changed from elif to if
        for item in data:
            if isinstance(item, (dict, list)):
                _transform_validators(item)


def _transform_options(data: Dict[str, Any]) -> None:
    """Transform options fields to match Pydantic model expectations."""
    if isinstance(data, dict):
        if 'options' in data and isinstance(data['options'], list):
            for option in data['options']:
                if isinstance(option, dict) and 'dependentControls' in option:
                    if isinstance(option['dependentControls'], list) and all(isinstance(item, dict) for item in option['dependentControls']):
                        option['dependentControls'] = option['dependentControls']
        for key, value in list(data.items()):
            if isinstance(value, (dict, list)):
                _transform_options(value)
    if isinstance(data, list):  # Changed from elif to if
        for item in data:
            if isinstance(item, (dict, list)):
                _transform_options(item)


class FormBuilder:
    def __init__(self):
        self.reset_form()
        self.expected_controls_count = 0  # Track expected number of controls
        self.current_form_id = None  # To track the current form being modified

    def reset_form(self):
        """Reset the form to initial state"""
        self.form = IForm(
            value=None,
            valid=None,
            get=None,
            formTitle="Untitled Form",
            saveBtnTitle=None,
            saveBtnFunction=None,
            resetBtnTitle=None,
            calculateBtnTitle=None,
            prevBtnTitle=None,
            themeFile="default_theme.json",
            formSections=[],
            class_=None
        )
        self.expected_controls_count = 0
        
    def set_expected_controls(self, count: int):
        """Set the expected number of controls to be added to the form"""
        self.expected_controls_count = count

    def add_section(self, sectionTitle: str) -> Dict:
        """Add a new section to the form."""
        # Check if section already exists
        for section in self.form.formSections:
            if section.sectionTitle.lower() == sectionTitle.lower():
                return {"status": "warning", "message": f"Section '{sectionTitle}' already exists"}
                
        section_id = f"section_{len(self.form.formSections)}"
        new_section = IFormSections(
            sectionTitle=sectionTitle,
            visible=True,
            apiEndpoint=None,
            controlTypeName=None,
            method=None,
            isVisible=True,
            formControls=[],
            visibleLabel=True,
            sectionButton=None,
            class_=None,
            toolTipText=None,
            urlDependentControls=None,
            urlPath=None,
            productFeaturesUrl=None
        )
        self.form.formSections.append(new_section)
        return {"status": "success", "section_id": section_id}

    def add_control(self, sectionTitle: str, controlType: str, label: str = None, name: str = None, required: bool = False, validation: Optional[Dict] = None) -> Dict:
        """
        Add a new control to a section or update an existing control's label or name.
        - If 'label' is provided, update the label but keep the name unchanged.
        - If 'name' is provided, update the name but keep the label unchanged.
        """
        for section in self.form.formSections:
            if section.sectionTitle == sectionTitle:
                # Check if a control with the same name or label already exists
                for existing_control in section.formControls:
                    if existing_control.name == name or existing_control.label.lower() == label.lower():
                        # Control exists: Update the label or name independently
                        if label is not None:
                            existing_control.label = label
                        if name is not None:
                            existing_control.name = name
                        return {"status": "success", "message": f"Updated control '{existing_control.name}'"}

                # Control does not exist: Add a new control
                new_name = name or to_camel_case(label)  # Use provided name or generate from label
                new_label = label or name  # Use provided label or default to name
                control = IFormControl(
                    name=new_name,
                    label=new_label,
                    visibleLabel=True,
                    type_=controlType,
                    validators=[IValidator(required=required, **(validation or {}))] if required or validation else None
                )
                section.formControls.append(control)
                return {"status": "success", "control_name": control.name}
        return {"error": f"Section '{sectionTitle}' not found"}

    def delete_section(self, sectionTitle: str) -> Dict:
        """Delete a section from the form."""
        for i, section in enumerate(self.form.formSections):
            if section.sectionTitle == sectionTitle:
                self.form.formSections.pop(i)
                return {"status": "success", "message": f"Section '{sectionTitle}' deleted"}
        return {"error": f"Section '{sectionTitle}' not found"}

    def delete_control(self, sectionTitle: str, controlName: str) -> Dict:
        """Delete a control from a section. Works with either control name or label."""
        for section in self.form.formSections:
            if section.sectionTitle == sectionTitle:
                for i, control in enumerate(section.formControls):
                    # Match by exact name or by label (case insensitive)
                    if control.name == controlName or control.label.lower() == controlName.lower():
                        section.formControls.pop(i)
                        return {"status": "success", "message": f"Control '{controlName}' deleted from section '{sectionTitle}'"}
                return {"error": f"Control '{controlName}' not found in section '{sectionTitle}'"}
        return {"error": f"Section '{sectionTitle}' not found"}

    def update_control_validation(self, sectionTitle: str, controlName: str, validation: Dict) -> Dict:
        """Update validation rules for a control."""
        # Find the section first
        target_section = None
        for section in self.form.formSections:
            if section.sectionTitle == sectionTitle:
                target_section = section
                break
        
        if not target_section:
            return {"error": f"Section '{sectionTitle}' not found"}
        
        # Find the control in that section
        target_control = None
        for control in target_section.formControls:
            # Match by exact name or by label (case insensitive)
            if control.name == controlName or control.label.lower() == controlName.lower():
                target_control = control
                break
        
        if not target_control:
            # Make the error message more helpful
            existing_controls = [f"{c.name} (label: {c.label})" for c in target_section.formControls]
            return {
                "error": f"Control '{controlName}' not found in section '{sectionTitle}'. Available controls: {existing_controls}"
            }
        
        # Create or update validators
        if not target_control.validators:
            target_control.validators = [IValidator(**validation)]
        else:
            # Update existing validator
            for validator in target_control.validators:
                for key, value in validation.items():
                    setattr(validator, key, value)
                    
        return {"status": "success", "message": f"Validation updated for control '{target_control.name}'"}

    def set_form_title(self, title: str) -> Dict:
        """Set the form title."""
        self.form.formTitle = title
        return {"status": "success", "form_title": title}

    def is_form_complete(self) -> bool:
        """Check if the form has all expected controls."""
        total_controls = sum(len(section.formControls) for section in self.form.formSections)
        
        # Check if we have at least one section with controls
        has_sufficient_structure = (
            len(self.form.formSections) > 0 and 
            any(len(section.formControls) > 0 for section in self.form.formSections)
        )
        
        # In case expected_controls_count is 0 (not properly set), use a default minimum
        min_expected = max(1, self.expected_controls_count)
        
        # Consider complete ONLY if we have all expected controls or we've gone through multiple iterations
        return has_sufficient_structure and total_controls >= min_expected

    def remove_duplicate_controls(self) -> Dict:
        """Remove duplicate controls that have the same label within the same section."""
        duplicates_removed = 0
        
        for section in self.form.formSections:
            # Track controls we've seen by their label (case insensitive)
            seen_labels = {}
            controls_to_keep = []
            
            for control in section.formControls:
                label_key = control.label.lower()
                
                if label_key not in seen_labels:
                    # First time seeing this label, keep it
                    seen_labels[label_key] = control
                    controls_to_keep.append(control)
                else:
                    # Duplicate found, skip it
                    duplicates_removed += 1
            
            # Replace the section's controls with the de-duplicated list
            section.formControls = controls_to_keep
        
        return {
            "status": "success", 
            "message": f"Removed {duplicates_removed} duplicate controls",
            "duplicates_removed": duplicates_removed
        }

    # Update the get_current_form method in FormBuilder class
    def get_current_form(self) -> Dict:
        """Return the current form state with null values removed."""
        try:
            form_data = self.form.model_dump()
            return clean_null_values(form_data)
        except AttributeError:
            form_data = self.form.dict()
            return clean_null_values(form_data)
    
    def load_form_data(self, form_data: Dict) -> Dict:
        """Load existing form data into the form builder."""
        try:
            # Step 1: Transform the JSON data to match Pydantic models
            transformed_data = transform_json_for_pydantic(form_data)

            # Step 2: Process the transformed data
            processed_data = {
                "value": None,
                "valid": None,
                "get": None,
                "formTitle": "Untitled Form",
                "saveBtnTitle": "Save",
                "resetBtnTitle": "Reset",
                "calculateBtnTitle": None,
                "prevBtnTitle": None,
                "themeFile": "default_theme.json",
                "formSections": [],
                "class_": None,
                "saveBtnFunction": None
            }

            # Update with the transformed values
            for key, value in transformed_data.items():
                if key == "formSections":
                    processed_sections = []
                    for section in value:
                        # Ensure section has required fields
                        section_data = {
                            "sectionTitle": section.get("sectionTitle", "Untitled Section"),
                            "visible": section.get("visible", True),
                            "apiEndpoint": section.get("apiEndpoint", None),
                            "controlTypeName": section.get("controlTypeName", None),
                            "method": section.get("method", None),
                            "isVisible": section.get("isVisible", True),
                            "formControls": [],
                            "visibleLabel": section.get("visibleLabel", True),
                            "sectionButton": section.get("sectionButton", None),
                            "class_": section.get("class_", None),
                            "toolTipText": section.get("toolTipText", None),
                            "urlDependentControls": section.get("urlDependentControls", None),
                            "urlPath": section.get("urlPath", None),
                            "productFeaturesUrl": section.get("productFeaturesUrl", None)
                        }
                        # Process controls in this section
                        if "formControls" in section and isinstance(section["formControls"], list):
                            for control in section["formControls"]:
                                control_data = {
                                    "name": control.get("name", f"control_{len(section_data['formControls'])}"),
                                    "label": control.get("label", "Untitled Control"),
                                    "visibleLabel": control.get("visibleLabel", True),
                                    "type_": control.get("type_", "text"),
                                    "validators": control.get("validators", None)
                                }
                                section_data["formControls"].append(IFormControl(**control_data))
                        processed_sections.append(IFormSections(**section_data))
                    processed_data["formSections"] = processed_sections
                else:
                    processed_data[key] = value

            # Create new form instance with the processed data
            self.form = IForm(**processed_data)
            total_controls = sum(len(section.formControls) for section in self.form.formSections)
            self.expected_controls_count = total_controls
            self.current_form_id = processed_data.get("id", None)
            return {
                "status": "success", 
                "message": f"Form loaded successfully with {total_controls} controls"
            }
        except Exception as e:
            print(f"Exception in load_form_data: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": f"Error loading form data: {str(e)}"}
# Initialize FormBuilder
builder = FormBuilder()

def get_form_templates_list() -> List[str]:
    """Get a list of available form template filenames."""
    templates_dir = Path("src/form_builder/form_templates")
    if not templates_dir.exists():
        templates_dir = Path("form_templates")  # Fallback path
    
    templates = []
    if templates_dir.exists():
        # Update to include both .json and .txt files
        templates = [f.name for f in templates_dir.glob("*.json")]
        templates.extend([f.name for f in templates_dir.glob("*.txt")])
    return templates

def load_form_template(template_name: str) -> Dict:
    """Load a form template from the form_templates directory."""
    templates_dir = Path("src/form_builder/form_templates")
    if not templates_dir.exists():
        templates_dir = Path("form_templates")  # Fallback path
    
    # Check if the template_name already has an extension
    if not template_name.endswith(('.json', '.txt')):
        # Try with .json extension first
        json_path = templates_dir / f"{template_name}.json"
        txt_path = templates_dir / f"{template_name}.txt"
        
        if json_path.exists():
            template_path = json_path
        elif txt_path.exists():
            template_path = txt_path
        else:
            raise FileNotFoundError(f"Template '{template_name}' not found with either .json or .txt extension")
    else:
        # Template name already has an extension
        template_path = templates_dir / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template '{template_name}' not found")
    
    with open(template_path, "r") as f:
        content = f.read()
        
        # Check if it's a .txt file and try to parse JSON from it
        if template_path.suffix.lower() == '.txt':
            try:
                template_data = json.loads(content)
            except json.JSONDecodeError:
                raise ValueError(f"The .txt file '{template_name}' does not contain valid JSON data")
        else:
            # For .json files, load directly
            template_data = json.loads(content)
    
    return template_data


# Define tools for the agent
tools = {
    "add_section": {
        "description": "Adds a new section to the form.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {"type": "string", "description": "Title of the section"}
            },
            "required": ["sectionTitle"]
        },
        "func": builder.add_section
    },
    "add_control": {
        "description": "Adds a new control to a section.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {"type": "string", "description": "Title of the section"},
                "controlType": {"type": "string", "description": "Type of the control (text, email, number, date, select, textarea, checkbox, radio)"},
                "label": {"type": "string", "description": "Label of the control"},
                "required": {"type": "boolean", "description": "Whether the control is required"},
                "validation": {"type": "object", "description": "Validation rules for the control"}
            },
            "required": ["sectionTitle", "controlType", "label"]
        },
        "func": builder.add_control
    },
    "delete_section": {
        "description": "Deletes a section from the form.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {"type": "string", "description": "Title of the section to delete"}
            },
            "required": ["sectionTitle"]
        },
        "func": builder.delete_section
    },
    "delete_control": {
        "description": "Deletes a control from a section.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {"type": "string", "description": "Title of the section"},
                "controlName": {"type": "string", "description": "Name or label of the control to delete"}
            },
            "required": ["sectionTitle", "controlName"]
        },
        "func": builder.delete_control
    },
    "update_control_validation": {
        "description": "Updates validation rules for a control.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {"type": "string", "description": "Title of the section"},
                "controlName": {"type": "string", "description": "Name or label of the control"},
                "validation": {"type": "object", "description": "Updated validation rules"}
            },
            "required": ["sectionTitle", "controlName", "validation"]
        },
        "func": builder.update_control_validation
    },
    "set_form_title": {
        "description": "Sets the title of the form.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title for the form"}
            },
            "required": ["title"]
        },
        "func": builder.set_form_title
    }
}


def extract_sections_and_controls(prompt: str) -> List[Dict]:
    """
    Dynamically extract sections, controls, and validation rules from the user's prompt.
    Example: "Create a form with sections Basic Information and Contact Details. Add Full Name (text, required, max length 50), Email (email, required), and Age (number, min value 18)."
    Returns: [
        {"section": "Basic Information", "controls": [{"field": "Full Name", "type": "text", "validations": {"required": True, "maxLength": 50}}]},
        {"section": "Contact Details", "controls": [{"field": "Email", "type": "email", "validations": {"required": True}}]}
    ]
    """
    section_patterns = r"\b(?:sections?|section titled)\s*([\w\s,]+)"
    sections = []
    section_match = re.search(section_patterns, prompt, re.IGNORECASE)
    if section_match:
        sections = [section.strip() for section in section_match.group(1).split(",")]

    # Default to a single section if none found
    if not sections:
        sections = ["Form Section"]

    extracted_data = []
    field_validation_keywords = {
        "required": r"\b(required)\b",
        "max_length": r"\b(max\s*length|maxlen)\s*(\d+)",
        "min_length": r"\b(min\s*length|minlen)\s*(\d+)",
        "email": r"\b(email)\b",
        "min_value": r"\b(min\s*value|minval)\s*(\d+)",
        "max_value": r"\b(max\s*value|maxval)\s*(\d+)",
        "pattern": r"\b(pattern)\s*([\w\W]+)",
    }

    # If we have multiple sections, try to distribute fields
    field_pattern = r"([\w\s]+)\s*\(([\w\s,]+)\)"
    field_matches = re.findall(field_pattern, prompt, re.IGNORECASE)
    
    total_controls = len(field_matches)
    builder.set_expected_controls(total_controls)  # Set the expected control count
    
    # Simple distribution - first field to first section, etc.
    section_idx = 0
    fields_by_section = {section: [] for section in sections}
    
    for field_name, field_details in field_matches:
        # Assign field to current section
        fields_by_section[sections[section_idx]].append((field_name, field_details))
        # Move to next section (with wrap-around)
        section_idx = (section_idx + 1) % len(sections)

    for section, fields in fields_by_section.items():
        fields_with_validations = []
        
        for field_name, field_details in fields:
            field_name = field_name.strip()
            validations = {}
            field_type = None

            # Infer field type from details
            if "text" in field_details.lower():
                field_type = "text"
            elif "email" in field_details.lower():
                field_type = "email"
            elif "number" in field_details.lower():
                field_type = "number"
            elif "date" in field_details.lower():
                field_type = "date"
            elif "select" in field_details.lower():
                field_type = "select"
            elif "textarea" in field_details.lower():
                field_type = "textarea"
            elif "checkbox" in field_details.lower():
                field_type = "checkbox"
            elif "radio" in field_details.lower():
                field_type = "radio"
            elif "file" in field_details.lower():
                field_type = "file"
            else:
                # Default to text if no type specified
                field_type = "text"

            # Extract validation rules
            for key, keyword_pattern in field_validation_keywords.items():
                validation_match = re.search(keyword_pattern, field_details, re.IGNORECASE)
                if validation_match:
                    if key in ["max_length", "min_length", "min_value", "max_value"]:
                        validations[key] = int(validation_match.group(2))
                    elif key in ["required", "email"]:
                        validations[key] = True
                    elif key == "pattern":
                        validations[key] = validation_match.group(2).strip()

            fields_with_validations.append({
                "field": field_name,
                "type": field_type,
                "validations": validations
            })

        if fields_with_validations:  # Only add sections with fields
            extracted_data.append({"section": section, "controls": fields_with_validations})

    return extracted_data


async def detect_request_type(prompt: str) -> bool:
    """Use Gemini to determine if the user wants to modify an existing form or create a new one."""
    current_form = builder.get_current_form()
    has_existing_form = (
        len(current_form.get("formSections", [])) > 0 and
        any(len(section.get("formControls", [])) > 0 for section in current_form.get("formSections", []))
    )
    
    if not has_existing_form:
        return False
    
    if builder.current_form_id is not None:
        return True
    
    modification_keywords = [
        "add", "change", "modify", "update", "delete", "remove", 
        "edit", "alter", "adjust", "revise", "include", "append"
    ]
    
    for keyword in modification_keywords:
        if re.search(r'\b' + keyword + r'\b', prompt, re.IGNORECASE):
            return True
    
    try:
        system_prompt = """
        You are a request classifier for a form builder application. 
        Your task is to determine whether the user's request is:
        
        1. Creating a NEW form from scratch, or
        2. MODIFYING an existing form
        
        Examine the request and respond with ONLY "NEW" or "MODIFY".
        """
        
        # Create chat and get response
        chat = model.start_chat()
        response = chat.send_message(
            f"{system_prompt}\n\nUser request: {prompt}",
            generation_config={"temperature": 0.1}
        )
        
        result = response.text.strip().upper()
        is_modification = "MODIFY" in result
        print(f"Request classification: {'MODIFY' if is_modification else 'NEW'}")
        return is_modification
    except Exception as e:
        print(f"Error classifying request: {e}")
        return False

# Define the workflow state type
class WorkflowState(TypedDict, total=False):
    prompt: str
    messages: List[Dict[str, Any]]
    iteration_count: int
    tool_calls: List[Dict[str, Any]]
    form_complete: bool
    extracted_data: List[Dict[str, Any]]
    results: List[Dict[str, Any]]
    form: Any
    is_modification: bool
    is_json_modification: bool  # New field
    successful_tools: int
    failed_tools: int


# Define the workflow
workflow = Graph()



@workflow.add_node
async def start_node(state: WorkflowState) -> WorkflowState:
    """Process the initial user input."""
    prompt = state["prompt"]
    is_json_modification = state.get("is_json_modification", False)
    
    # If it's not a JSON modification, use the existing logic
    if not is_json_modification:
        # Use LLM to determine if this is a modification request or a new form request
        is_modification = await detect_request_type(prompt)
        
        # If it's a new form request, reset the form builder
        if not is_modification:
            builder.reset_form()
            # Dynamically extract sections and controls
            extracted_data = extract_sections_and_controls(prompt)
        else:
            # For modification requests, we don't reset and don't extract new form structure
            extracted_data = []
    else:
        # If it's a JSON modification request, we've already loaded the form
        # Mark it as a modification request
        is_modification = True
        extracted_data = []

    system_prompt = ""
    
    if is_modification:
        # Prompt for modifying an existing form
        current_form = builder.get_current_form()
        sections_info = []
        
        for idx, section in enumerate(current_form.get("formSections", [])):
            controls_info = []
            for control in section.get("formControls", []):
                validators = control.get("validators", [])
                validation_info = ""
                if validators:
                    validation_details = []
                    for validator in validators:
                        for key, value in validator.items():
                            if key != "type_":  # Skip type field
                                validation_details.append(f"{key}: {value}")
                    if validation_details:
                        validation_info = f" with validation ({', '.join(validation_details)})"
                
                controls_info.append(f"{control.get('name')}: {control.get('label')} ({control.get('type_')}){validation_info}")
            
            if controls_info:
                section_detail = f"Section {idx+1}: {section.get('sectionTitle')}\n"
                section_detail += "\n".join([f"  - {control}" for control in controls_info])
                sections_info.append(section_detail)
            else:
                # Show empty sections too
                section_detail = f"Section {idx+1}: {section.get('sectionTitle')} (empty)"
                sections_info.append(section_detail)
        
        form_details = "\n\n".join(sections_info)
        
        source_info = "from JSON input" if is_json_modification else "from existing session"
        
        # Parse fields to add from the prompt for modification
        fields_to_add = []
        field_pattern = r"([\w\s]+)\s*\(([\w\s,]+)\)"
        field_matches = re.findall(field_pattern, prompt, re.IGNORECASE)
        
        if field_matches:
            # Extract target section from prompt
            section_pattern = r"to\s+(?:the\s+)?(?:[\w\s]*\s+)?(?:section|tab)\s+['\"]?([\w\s]+)['\"]?"
            section_match = re.search(section_pattern, prompt, re.IGNORECASE)
            target_section = section_match.group(1) if section_match else None
            
            fields_info = ""
            if target_section:
                fields_info = f"Fields to add to section '{target_section}':\n"
                for field_name, field_details in field_matches:
                    fields_info += f"- {field_name.strip()} ({field_details.strip()})\n"
        
        system_prompt = f"""
        You are a form modification assistant. Your task is to modify an existing form based on user requirements.
        
        The form was loaded {source_info}.

        Current form structure:
        Title: {current_form.get('formTitle')}
        {form_details}
        
        User's request: "{prompt}"

        IMPORTANT: You MUST respond with ONLY a JSON array of tool calls to make the requested modifications.
        Each tool call must be an object with "name" and "parameters" fields.

        Available tools:
        1. add_section - Add a new section to the form
        2. add_control - Add a new control to a section
        3. delete_section - Delete a section from the form
        4. delete_control - Delete a control from a section
        5. update_control_validation - Update validation rules for a control
        6. set_form_title - Change the form title

        When modifying controls:
        - For delete_control and update_control_validation, you can use either the control's name or label
        - For add_control, ensure the label doesn't already exist in that section
        - Make sure to specify all required parameters for each tool call
        - Be precise with section titles and control names/labels

        When using add_control:
        - If the control already exists (by name or label):
        - Only the requested field (label or name) will be updated, and the other field will remain unchanged.
        - If the control does not exist, a new control will be added with the name set to the camelCase version of the label.
        Example response format:
        [
            {{
                "name": "add_control",
                "parameters": {{
                    "sectionTitle": "Address Information",
                    "controlType": "text",
                    "label": "Street Address",
                    "required": true,
                    "validation": {{}}
                }}
            }},
            {{
                "name": "add_control",
                "parameters": {{
                    "sectionTitle": "Address Information",
                    "controlType": "text",
                    "label": "City",
                    "required": true,
                    "validation": {{}}
                }}
            }}
        ]

        DO NOT return a description of the form or explain what you're doing.
        ONLY return the array of tool calls needed to implement the user's requested changes.
        """
    else:
        # Keep the existing prompt for creating a new form
        system_prompt = f"""
        You are a form generation assistant. Your task is to create forms based on user requirements.
        You have access to these tools:
        1. add_section - Use this to add a new section to the form.
        2. add_control - Use this to add form fields to sections.
        3. set_form_title - Use this to set a descriptive title for the form.

        Follow these steps for every form request:
        1. First, set an appropriate form title using set_form_title.
        2. Create appropriate sections using add_section.
        3. Add ALL relevant controls to each section using add_control.

        For control types, choose from: text, email, number, date, select, textarea, checkbox, radio.
        Ensure that ALL fields mentioned in the user's request are included in the form.

        Sections and fields to include: {json.dumps(extracted_data)}

        IMPORTANT: You MUST respond with ONLY a JSON array of tool calls.
        Each tool call must be an object with "name" and "parameters" fields.

        Example response format:
        [
            {{
                "name": "set_form_title", 
                "parameters": {{"title": "Personal Information Form"}}
            }},
            {{
                "name": "add_section",
                "parameters": {{"sectionTitle": "Basic Information"}}
            }},
            {{
                "name": "add_control",
                "parameters": {{
                    "sectionTitle": "Basic Information",
                    "controlType": "text",
                    "label": "Full Name",
                    "required": true,
                    "validation": {{"maxLength": 50}}
                }}
            }}
        ]

        Make sure to create a complete and usable form with ALL necessary fields and validations based on the user's request.
        DO NOT explain what you're doing, ONLY return the array of tool calls.
        """

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "iteration_count": 0,
        "extracted_data": extracted_data,
        "is_modification": is_modification,
        "is_json_modification": is_json_modification,
        "successful_tools": 0,
        "failed_tools": 0
    }


# Improved tool calls extraction function
def extract_tool_calls_from_message(content: str) -> List[Dict]:
    """Extract tool calls from LLM messages even when formatted poorly."""
    tool_calls = []
    
    # First try JSON parsing the entire content
    try:
        # Clean up the content to ensure it's valid JSON
        # Remove markdown code block indicators if present
        cleaned_content = re.sub(r'```(?:json)?\s*([\s\S]*?)\s*```', r'\1', content)
        cleaned_content = cleaned_content.strip()
        
        # Check if the content is already wrapped in square brackets
        if not (cleaned_content.startswith('[') and cleaned_content.endswith(']')):
            # Try to extract JSON from text explanation
            json_pattern = r'\[\s*{[\s\S]*}\s*\]'
            json_match = re.search(json_pattern, cleaned_content)
            if json_match:
                cleaned_content = json_match.group(0)
        
        # Try parsing as JSON
        parsed_content = json.loads(cleaned_content)
        
        # Handle different response formats
        if isinstance(parsed_content, list):
            # Direct array of tool calls, which is what we want
            return [item for item in parsed_content if isinstance(item, dict) and "name" in item and "parameters" in item]
        elif isinstance(parsed_content, dict):
            # Single tool call as a dict
            if "name" in parsed_content and "parameters" in parsed_content:
                return [parsed_content]
    except Exception as e:
        print(f"Initial JSON parsing failed: {e}")
        # Continue to regex method
    
    # If that failed, try regex-based parsing for individual tool calls
    try:
        # Look for patterns like: {"name": "tool_name", "parameters": {...}}
        # First fix up common JSON formatting issues
        normalized_content = re.sub(r'([{,])\s*([a-zA-Z_]+):', r'\1"\2":', content)
        # Then try to find tool-like structures
        tool_pattern = r'{"name":\s*"([^"]+)",\s*"parameters":\s*({[^{}]*(?:{[^{}]*}[^{}]*)*})'
        matches = re.findall(tool_pattern, normalized_content)
        
        for name, params_str in matches:
            try:
                # Fix parameters JSON if needed
                params_str = re.sub(r'([{,])\s*([a-zA-Z_]+):', r'\1"\2":', params_str)
                params = json.loads(params_str)
                tool_calls.append({
                    "name": name,
                    "parameters": params
                })
            except Exception as params_e:
                print(f"Failed to parse parameters for {name}: {params_e}")
                continue
    except Exception as regex_e:
        print(f"Regex extraction failed: {regex_e}")
    
    return tool_calls


@workflow.add_node
def llm_node(state: WorkflowState) -> WorkflowState:
    """Call the LLM to determine actions."""
    messages = state["messages"]
    iteration_count = state.get("iteration_count", 0)
    extracted_data = state.get("extracted_data", [])
    is_modification = state.get("is_modification", False)
    successful_tools = state.get("successful_tools", 0)
    failed_tools = state.get("failed_tools", 0)

    # Prevent infinite loops
    if iteration_count >= 20:
        return {
            "messages": messages,
            "iteration_count": iteration_count,
            "tool_calls": [],
            "form_complete": True,
            "is_modification": is_modification,
            "successful_tools": successful_tools,
            "failed_tools": failed_tools
        }

    try:
        # Convert messages to Gemini format
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            # Map roles to Gemini format
            if role == "system":
                # Prepend system message to user message
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": "[System Instructions]\n" + msg["content"]}]
                })
            elif role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": msg["content"]}]
                })
            elif role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": msg["content"]}]
                })
            elif role == "tool":
                # Convert tool results to user message
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": "[Tool Results]\n" + msg["content"]}]
                })

        # Create chat session
        chat = model.start_chat(history=gemini_messages)
        
        # Generate response
        response = chat.send_message(
            messages[-1]["content"],
            generation_config={"temperature": 0.5}
        )
        
        content = response.text
        print(f"Gemini Response: {content}")
        
        # Process the response the same way as before
        tool_calls = extract_tool_calls_from_message(content)
        print(f"Initial Parsed Tool Calls: {tool_calls}")
        
        # Validate extracted tool calls
        valid_tool_calls = []
        seen_operations = set()
        
        for call in tool_calls:
            if isinstance(call, dict) and "name" in call and "parameters" in call:
                if call["name"] in tools:
                    # Create operation signature to detect duplicates
                    if call["name"] == "add_control":
                        section = call["parameters"].get("sectionTitle", "")
                        label = call["parameters"].get("label", "")
                        op_signature = f"add_control:{section}:{label}"
                    elif call["name"] in ["delete_control", "update_control_validation"]:
                        section = call["parameters"].get("sectionTitle", "")
                        control = call["parameters"].get("controlName", "")
                        op_signature = f"{call['name']}:{section}:{control}"
                    else:
                        op_signature = f"{call['name']}:{json.dumps(call['parameters'])}"
                    
                    if op_signature not in seen_operations:
                        valid_tool_calls.append(call)
                        seen_operations.add(op_signature)
        
        tool_calls = valid_tool_calls
        print(f"Validated Tool Calls: {tool_calls}")
        
        if is_modification:
            form_complete = (len(valid_tool_calls) == 0 and iteration_count > 1) or iteration_count >= 5
        else: 
            form_complete = builder.is_form_complete()

        return {
            "messages": messages + [{"role": "assistant", "content": content}],
            "tool_calls": valid_tool_calls, 
            "iteration_count": iteration_count + 1,
            "form_complete": form_complete,
            "is_modification": is_modification,
            "retry_parsing": False,
            "successful_tools": successful_tools,
            "failed_tools": failed_tools
        }
    except Exception as e:
        print(f"Error calling Gemini: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

@workflow.add_node
def tool_node(state: WorkflowState) -> WorkflowState:
    """Execute the tools."""
    tool_calls = state["tool_calls"]
    messages = state["messages"]
    results = []
    is_modification = state.get("is_modification", False)
    iteration_count = state.get("iteration_count", 0)
    successful_tools = state.get("successful_tools", 0)
    failed_tools = state.get("failed_tools", 0)

    # Process all tool calls in this batch
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tools:
            try:
                print(f"Executing tool '{tool_name}' with parameters: {tool_call['parameters']}")
                result = tools[tool_name]["func"](**tool_call["parameters"])
                print(f"Executed tool '{tool_name}' with result: {result}")
                
                if "status" in result and result["status"] in ["success", "warning"]:
                    successful_tools += 1
                
                results.append({
                    "tool_call_id": tool_call.get("id", ""),
                    "name": tool_name,
                    "result": result
                })
            except Exception as e:
                print(f"Error executing tool '{tool_name}': {e}")
                failed_tools += 1
                error_message = str(e)
                if "sectionTitle" in str(e).lower():
                    error_message = f"Missing or invalid sectionTitle parameter. Available sections: {[s.sectionTitle for s in builder.form.formSections]}"
                elif "controlName" in str(e).lower():
                    error_message = f"Missing or invalid controlName parameter. For this tool, you can use either the control's name (e.g., 'control_0') or its label."
                
                results.append({
                    "tool_call_id": tool_call.get("id", ""),
                    "name": tool_name,
                    "error": error_message
                })

    total_controls = sum(len(section.formControls) for section in builder.form.formSections)
    expected_controls = builder.expected_controls_count
    print(f"Progress: {total_controls}/{expected_controls} controls added")
    
    # MODIFIED LOGIC HERE - For modifications, only complete when we've run out of tool calls to process
    # or reached iteration limit (giving LLM more time to add everything)
    if is_modification:
        # For modifications, we're done if:
        # 1. We've processed all expected tool calls from the current LLM response, AND
        #    (We've had at least one successful operation OR we've hit the iteration limit)
        form_complete = (len(tool_calls) == 0 or iteration_count >= 5) and (successful_tools > 0 or iteration_count >= 5)
    else:
        # For new forms, keep the existing logic
        form_complete = builder.is_form_complete() and (successful_tools > 0 or iteration_count >= 5)
    
    print(f"Form complete: {form_complete} (Success: {successful_tools}, Failed: {failed_tools})")

    # Return the updated form state after processing tool calls
    updated_form = builder.get_current_form()

    # If all tools failed, provide clearer guidance
    if failed_tools > 0 and successful_tools == 0:
        available_sections = [s.sectionTitle for s in builder.form.formSections]
        controls_by_section = {}
        for section in builder.form.formSections:
            controls_by_section[section.sectionTitle] = [
                f"{c.name} (label: '{c.label}', type: {c.type_})" for c in section.formControls
            ]
        
        context_message = {
            "role": "user",
            "content": f"""
All tool calls failed. Please correct your approach:
1. Available sections: {available_sections}
2. Controls in each section: {json.dumps(controls_by_section, indent=2)}
3. For delete_control and update_control_validation, you can use either the control's name or label
4. For add_control, ensure the label doesn't already exist in that section
5. For update_control_validation, provide complete validation rules

Try again with EXACT section names and proper parameters.
"""
        }
        messages.append(context_message)

    return {
        "messages": messages + [{
            "role": "tool",
            "content": json.dumps(results),
            "tool_call_id": tool_call.get("id", "") if tool_calls else ""
        }],
        "results": results,
        "form_complete": form_complete,
        "is_modification": is_modification,
        "iteration_count": iteration_count,
        "updated_form": updated_form,
        "successful_tools": successful_tools,
        "failed_tools": failed_tools
    }

@workflow.add_node
def end_node(state: WorkflowState) -> WorkflowState:
    """Finalize the form generation and clean up any duplicates."""
    # Remove any duplicate controls that might have been created
    dedupe_result = builder.remove_duplicate_controls()
    print(f"Deduplication result: {dedupe_result}")
    
    # Get the final form state
    current_form = builder.get_current_form()
    
    return {
        "form": current_form,
        "updated_form": current_form,
        "is_modification": state.get("is_modification", False),
        "successful_tools": state.get("successful_tools", 0),
        "failed_tools": state.get("failed_tools", 0),
        "deduplication": dedupe_result
    }


# Define the workflow edges
workflow.set_entry_point("start_node")
workflow.add_edge("start_node", "llm_node")

# After LLM node, decide whether to use tools or end
workflow.add_conditional_edges(
    "llm_node",
    lambda state: (
        state.get("form_complete", False) or 
        (not state.get("tool_calls") and not state.get("retry_parsing", False))
    ),
    {
        True: "end_node",
        False: "tool_node"
    }
)

# After tool execution, decide whether to continue or end
workflow.add_conditional_edges(
    "tool_node",
    lambda state: state.get("form_complete", False),
    {
        True: "end_node",
        False: "llm_node"
    }
)

workflow.add_edge("end_node", END)

# Compile the workflow
app_workflow = workflow.compile()



# Now, add a new endpoint to the FastAPI app to accept JSON form data
@app.post("/load-json-form")
async def load_json_form(form_data: Dict) -> Dict:
    """Load a form from JSON data."""
    try:
        result = builder.load_form_data(form_data)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Error loading form data"))
        return {
            "status": "success",
            "result": result,
            "form": builder.get_current_form()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# API Endpoints
class FormRequestModel(BaseModel):
    prompt: str

class FormModificationRequest(BaseModel):
    prompt: str
    form_json: Optional[Dict] = None
    


@app.post("/generate-form")
async def generate_form(request: Union[FormRequestModel, FormModificationRequest, TemplateRequestModel]) -> Dict:
    """Generate or modify a form based on the user's prompt, optional JSON input, or template."""
    try:
        # Check what type of request we're handling
        form_json = getattr(request, "form_json", None)
        template_name = getattr(request, "template_name", None)
        
        # If template name is provided, load it first
        if template_name:
            try:
                template_data = load_form_template(template_name)
                
                # Skip validation - let the builder handle it
                load_result = builder.load_form_data(template_data)
                if load_result.get("status") == "error":
                    return {
                        "status": "error", 
                        "message": load_result.get("message", "Error loading template")
                    }
                
                # Mark this as a template modification request
                is_json_modification = True
                print(f"Loaded template '{template_name}' with result: {load_result}")
                print(f"Current form structure: {builder.get_current_form()}")
                
            except FileNotFoundError:
                return {
                    "status": "error",
                    "message": f"Template '{template_name}' not found"
                }
        # If form JSON is provided, handle as before
        elif form_json:
            # Skip validation - let the builder handle it
            load_result = builder.load_form_data(form_json)
            if load_result.get("status") == "error":
                return {
                    "status": "error", 
                    "message": load_result.get("message", "Error loading form data")
                }
            
            # Mark this as a modification request
            is_json_modification = True
            print(f"Loaded form data with result: {load_result}")
            print(f"Current form structure: {builder.get_current_form()}")
        else:
            is_json_modification = False
        
        # Execute the workflow with the config
        config = RunnableConfig(recursion_limit=50)
        
        # Pass additional context to the workflow
        result = await app_workflow.ainvoke({
            "prompt": request.prompt,
            "is_json_modification": is_json_modification
        }, config=config)
        
        # Always return the current state of the form from the builder
        current_form = builder.get_current_form()
        
        # Add metadata about the request type
        request_type = "template_modification" if template_name else (
            "json_modification" if form_json else (
                "modification" if result.get("is_modification", False) else "new_form"
            )
        )
        
        return {
            "form": current_form,
            "request_type": request_type,
            "template": template_name if template_name else None,
            "successful_operations": result.get("successful_tools", 0),
            "failed_operations": result.get("failed_tools", 0),
            "deduplication": result.get("deduplication", {})
        }
    except Exception as e:
        print(f"Error in generate_form: {str(e)}")  # Log the error
        traceback.print_exc()  # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/add-section")
async def add_section(title: str) -> Dict:
    """Manually add a section to the form."""
    try:
        result = builder.add_section(sectionTitle=title)
        return {
            "status": "success", 
            "result": result,
            "form": builder.get_current_form()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-control")
async def add_control(
    section_title: str,
    control_type: str,
    label: str,
    required: bool = False,
    validation: Optional[Dict] = None
) -> Dict:
    """Manually add a control to a section."""
    try:
        result = builder.add_control(
            sectionTitle=section_title,
            controlType=control_type,
            label=label,
            required=required,
            validation=validation
        )
        return {
            "status": "success", 
            "result": result,
            "form": builder.get_current_form()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-section")
async def delete_section(section_title: str) -> Dict:
    """Manually delete a section from the form."""
    try:
        result = builder.delete_section(sectionTitle=section_title)
        return {
            "status": "success", 
            "result": result,
            "form": builder.get_current_form()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-control")
async def delete_control(section_title: str, control_name: str) -> Dict:
    """Manually delete a control from a section."""
    try:
        result = builder.delete_control(
            sectionTitle=section_title,
            controlName=control_name
        )
        return {
            "status": "success", 
            "result": result,
            "form": builder.get_current_form()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-validation")
async def update_validation(section_title: str, control_name: str, validation: Dict) -> Dict:
    """Manually update validation rules for a control."""
    try:
        result = builder.update_control_validation(
            sectionTitle=section_title,
            controlName=control_name,
            validation=validation
        )
        return {
            "status": "success", 
            "result": result,
            "form": builder.get_current_form()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/set-title")
async def set_title(title: str) -> Dict:
    """Manually set the form title."""
    try:
        result = builder.set_form_title(title=title)
        return {
            "status": "success", 
            "result": result,
            "form": builder.get_current_form()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/get-form")
async def get_form() -> Dict:
    """Get the current form state."""
    try:
        return builder.get_current_form()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/reset-form")
async def reset_form() -> Dict:
    """Reset the form to initial state."""
    try:
        builder.reset_form()
        return {"status": "success", "message": "Form reset to initial state"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/remove-duplicates")
async def remove_duplicates() -> Dict:
    """Remove duplicate controls from the form."""
    try:
        result = builder.remove_duplicate_controls()
        return {
            "status": "success",
            "result": result,
            "form": builder.get_current_form()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/form-templates")
async def get_templates() -> Dict:
    """Get list of available form templates."""
    try:
        templates = get_form_templates_list()
        return {
            "status": "success",
            "templates": templates
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-template")
async def load_template(template_name: str) -> Dict:
    """Load a form template by name."""
    try:
        template_data = load_form_template(template_name)
        # Simply pass the template data to the builder without validation
        # The builder should handle validation itself
        result = builder.load_form_data(template_data)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message", "Error loading template"))
        
        return {
            "status": "success", 
            "template": template_name,
            "result": result,
            "form": builder.get_current_form()
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-template")
async def save_template(template_name: str) -> Dict:
    """Save the current form as a template."""
    try:
        templates_dir = Path("src/form_builder/form_templates")
        if not templates_dir.exists():
            templates_dir = Path("form_templates")  # Fallback path
            
        if not templates_dir.exists():
            templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Preserve the extension if provided, otherwise default to .json
        if not template_name.endswith(('.json', '.txt')):
            template_name = f"{template_name}.json"  # Default to JSON
            
        template_path = templates_dir / template_name
        
        # Get current form data
        form_data = builder.get_current_form()
        
        # Save to file - format based on extension
        with open(template_path, "w") as f:
            if template_name.endswith('.json'):
                json.dump(form_data, f, indent=2)
            else:  # .txt file
                # Still save as JSON format but in a .txt file
                json_content = json.dumps(form_data, indent=2)
                f.write(json_content)
            
        return {
            "status": "success",
            "message": f"Form saved as template '{template_name}'",
            "template_path": str(template_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)