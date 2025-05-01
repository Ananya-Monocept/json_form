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
from form_models import (
    IForm,
    IFormSections,
    IFormControl,
    IValidator,
    IRadioOption,
    ISelectCheckboxOption,
    IImage,
    IAdditionalQuestion,
    IAdditionalQuestionOption,
)
from langchain_core.runnables.config import RunnableConfig
from fastapi.middleware.cors import CORSMiddleware
import traceback
from copy import deepcopy

load_dotenv()

app = FastAPI(title="Form Builder API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

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

def _transform_dependent_controls(data: Dict) -> None:
    """
    Transform dependentControls fields in the input data.
    - Converts lists of strings into lists of dictionaries with 'name' and 'visibility' fields.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "dependentControls" and isinstance(value, list):
                transformed_dependent_controls = []
                for dep_control in value:
                    if isinstance(dep_control, str):
                        transformed_dependent_controls.append(
                            {"name": dep_control, "visibility": True}
                        )
                    elif isinstance(dep_control, dict):
                        if not all(
                            key in dep_control for key in ["name", "visibility"]
                        ):
                            raise ValueError(
                                f"Invalid dependent control: {dep_control}. Each dependent control must have 'name' and 'visibility' fields."
                            )
                        transformed_dependent_controls.append(dep_control)
                    else:
                        raise ValueError(
                            f"Invalid dependent control: {dep_control}. Expected a string or dictionary."
                        )
                data[key] = transformed_dependent_controls
            elif isinstance(value, (dict, list)):
                _transform_dependent_controls(value)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _transform_dependent_controls(item)

def transform_json_for_pydantic(json_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform the JSON structure to match the Pydantic models without modifying the models.
    This function now supports multiple JSON structures:
    1. Existing structure (`formSections` directly under root).
    2. New structure nested under `data → jsonFormData`.
    3. Structure with `form → sections → fields`.
    4. Directly provided `formSections` at the root level.
    """
    transformed_data = {}
    try:
        form_data = (
            json_data.get("data", {}).get("data", {}).get("jsonFormData", {})
            or json_data.get("data", {}).get("jsonFormData", {})
            or json_data.get("form", {})
            or {}
        )
        if not form_data and "formSections" in json_data:
            form_data = {"formSections": json_data["formSections"]}
        if not form_data:
            raise ValueError(
                f"Could not find valid form data. Available top-level keys: {list(json_data.keys())}"
            )
        transformed_data = deepcopy(form_data)
        if "sections" in transformed_data:
            transformed_data["formSections"] = []
            for section in transformed_data.pop("sections", []):
                transformed_section = {
                    "sectionTitle": section.get("sectionName", "Untitled Section"),
                    "sectionId": section.get("sectionId", None),
                    "visibleLabel": True,
                    "visible": True,
                    "class_": None,
                    "formControls": [],
                }
                seen_names = set()
                for field in section.get("fields", []):
                    if (
                        field.get("visible", True) is False
                        or field.get("disabled", False) is True
                    ):
                        continue
                    label = field.get(
                        "label", field.get("fieldName", "Unnamed Control")
                    )
                    name = field.get("fieldId", "")
                    if not name:
                        name = to_camel_case(label)
                    if name in seen_names:
                        name = f"{name}_duplicate"
                    seen_names.add(name)
                    transformed_field = {
                        "name": name,
                        "label": label,
                        "visibleLabel": True,
                        "type_": field.get("type", "text"),
                        "validators": [],
                        "visibilityRules": field.get("visibilityRules", None),
                    }
                    for validator in field.get("validators", []):
                        transformed_validator = {}
                        if validator.get("type") == "required":
                            transformed_validator["validatorName"] = "required"
                            transformed_validator["required"] = True
                            transformed_validator["message"] = validator.get(
                                "message", "This field is required."
                            )
                        elif validator.get("type") == "pattern":
                            transformed_validator["validatorName"] = "pattern"
                            transformed_validator["pattern"] = validator.get(
                                "value", ""
                            )
                            transformed_validator["message"] = validator.get(
                                "message", "Invalid format."
                            )
                        elif validator.get("type") == "minlength":
                            transformed_validator["validatorName"] = "minLength"
                            transformed_validator["minLength"] = validator.get(
                                "value", 0
                            )
                            transformed_validator["message"] = validator.get(
                                "message", "Input too short."
                            )
                        elif validator.get("type") == "maxlength":
                            transformed_validator["validatorName"] = "maxLength"
                            transformed_validator["maxLength"] = validator.get(
                                "value", 0
                            )
                            transformed_validator["message"] = validator.get(
                                "message", "Input too long."
                            )
                        if transformed_validator:
                            transformed_field["validators"].append(
                                transformed_validator
                            )
                    if "options" in field:
                        transformed_field["options"] = [
                            {"value": option.get("value"), "label": option.get("label")}
                            for option in field["options"]
                        ]
                    transformed_section["formControls"].append(transformed_field)
                transformed_data["formSections"].append(transformed_section)
        if "formSections" in transformed_data:
            for section in transformed_data["formSections"]:
                if "class" in section and "class_" not in section:
                    section["class_"] = section.pop("class")
                seen_names = set()
                for control in section.get("formControls", []):
                    if (
                        control.get("visible", True) is False
                        or control.get("disabled", False) is True
                    ):
                        continue
                    label = control.get("label", "Unnamed Control")
                    name = control.get("name", "")
                    if not name:
                        name = to_camel_case(label)
                    if name in seen_names:
                        name = f"{name}_duplicate"
                    seen_names.add(name)
                    control["name"] = name
                    if "class" in control and "class_" not in control:
                        control["class_"] = control.pop("class")
                    for validator in control.get("validators", []):
                        if "minlength" in validator:
                            validator["minLength"] = validator.pop("minlength")
                        if "maxlength" in validator:
                            validator["maxLength"] = validator.pop("maxlength")
        _transform_class_fields(transformed_data)
        _transform_type_fields(transformed_data)
        _transform_validators(transformed_data)
        _transform_options(transformed_data)
        return transformed_data
    except Exception as e:
        raise ValueError(f"Error transforming JSON: {str(e)}")

def to_camel_case(label: str) -> str:
    """Convert a label to camelCase."""
    words = label.strip().lower().split()
    if not words:
        return ""
    return words[0] + "".join(word.capitalize() for word in words[1:])

def _transform_class_fields(data: Dict[str, Any]) -> None:
    """Transform 'class' fields to 'class_' for Pydantic compatibility."""
    if isinstance(data, dict):
        if "class" in data:
            data["class_"] = data.pop("class")
        for key, value in list(data.items()):
            if isinstance(value, (dict, list)):
                _transform_class_fields(value)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _transform_class_fields(item)

def _transform_type_fields(data: Dict[str, Any]) -> None:
    """Transform 'type' fields to 'type_' for Pydantic compatibility."""
    if isinstance(data, dict):
        if "type" in data:
            data["type_"] = data.pop("type")
        for key, value in list(data.items()):
            if isinstance(value, (dict, list)):
                _transform_type_fields(value)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _transform_type_fields(item)

def _transform_validators(data: Dict[str, Any]) -> None:
    """Transform validator fields to match Pydantic model expectations."""
    if isinstance(data, dict):
        if "validators" in data and isinstance(data["validators"], list):
            for validator in data["validators"]:
                if "minlength" in validator:
                    validator["minLength"] = validator.pop("minlength")
                if "maxlength" in validator:
                    validator["maxLength"] = validator.pop("maxlength")
        for key, value in list(data.items()):
            if isinstance(value, (dict, list)):
                _transform_validators(value)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _transform_validators(item)

def _transform_options(data: Dict[str, Any]) -> None:
    """
    Transform options fields to match Pydantic model expectations.
    """
    if isinstance(data, dict):
        if "options" in data and isinstance(data["options"], list):
            for option in data["options"]:
                if isinstance(option, dict) and "dependentControls" in option:
                    if isinstance(option["dependentControls"], list):
                        transformed_dependent_controls = []
                        for dep_control in option["dependentControls"]:
                            if (
                                not isinstance(dep_control, dict)
                                or "name" not in dep_control
                            ):
                                raise ValueError(
                                    f"Invalid dependent control: {dep_control}. Each dependent control must have a 'name' field."
                                )
                            transformed_dependent_controls.append(dep_control["name"])
                        option["dependentControls"] = transformed_dependent_controls
        for key, value in list(data.items()):
            if isinstance(value, (dict, list)):
                _transform_options(value)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                _transform_options(item)

class FormBuilder:
    def __init__(self):
        self.reset_form()
        self.expected_controls_count = 0
        self.current_form_id = None

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
            class_=None,
        )
        self.expected_controls_count = 0

    def set_expected_controls(self, count: int):
        """Set the expected number of controls to be added to the form"""
        self.expected_controls_count = count

    def add_section(self, sectionTitle: str) -> Dict:
        """Add a new section to the form."""
        normalized_title = sectionTitle.strip()

        existing_section = None
        for section in self.form.formSections:
            if section.sectionTitle.lower() == normalized_title.lower():
                existing_section = section
                break
        if existing_section:
            return {
                "status": "success",
                "message": f"Using existing section '{sectionTitle}'",
                "section": existing_section,
            }
        section_id = f"section_{len(self.form.formSections)}"
        new_section = IFormSections(
            sectionTitle=normalized_title,
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
            productFeaturesUrl=None,
        )
        self.form.formSections.append(new_section)
        return {"status": "success", "section_id": section_id}

    def add_control(
        self,
        sectionTitle: str,
        controlType: str,
        label: str = None,
        name: str = None,
        required: bool = False,
        validation: Optional[Dict] = None,
        dependentControls: Optional[List[Dict[str, Any]]] = None,
        radioOptions: Optional[List[Dict[str, Any]]] = None,
        visibilityRules: Optional[Dict[str, Any]] = None,
        visible: Optional[bool] = True,
    ) -> Dict:
        """
        Add a new control to a section or update an existing control.
        Args:
            sectionTitle (str): Title of the section to add the control to
            controlType (str): Type of control (text, radio, etc.)
            label (str, optional): Label for the control
            name (str, optional): Name identifier for the control
            required (bool): Whether the control is required
            validation (Dict, optional): Validation rules
            dependentControls (List[Dict], optional): List of dependent controls
            radioOptions (List[Dict], optional): Options for radio controls
            visibilityRules (Dict, optional): Rules for conditional visibility
            visible (bool, optional): Initial visibility state
        """
        if not sectionTitle:
            return {"error": "Section title is required"}
        if not controlType:
            return {"error": "Control type is required"}
        if not label and not name:
            return {"error": "Either label or name must be provided"}
        normalized_title = sectionTitle.strip()
        target_section = None
        for section in self.form.formSections:
            if section.sectionTitle.lower() == normalized_title.lower():
                target_section = section
                break
        if not target_section:
            section_result = self.add_section(normalized_title)
            if section_result["status"] == "success":
                target_section = self.form.formSections[-1]
            else:
                return {"error": f"Failed to create section '{normalized_title}'"}
        try:
            new_name = name or to_camel_case(label)
            new_label = label or name
            for existing_control in target_section.formControls:
                if (existing_control.name and existing_control.name == new_name) or (
                    existing_control.label
                    and existing_control.label.lower() == new_label.lower()
                ):
                    if label is not None:
                        existing_control.label = new_label
                    if name is not None:
                        existing_control.name = new_name
                    if dependentControls is not None:
                        existing_control.dependentControls = dependentControls
                        self._update_dependent_controls(
                            target_section, existing_control.name, dependentControls
                        )
                    if radioOptions is not None:
                        existing_control.radioOptions = [
                            IRadioOption(
                                name=opt["name"],
                                label=opt["label"],
                                value=opt["value"],
                                selected=opt.get("selected", False),
                                visible=opt.get("visible", True),
                                dependentControls=[
                                    {"name": dc["name"], "visibility": dc["visibility"]}
                                    for dc in opt.get("dependentControls", [])
                                ],
                            )
                            for opt in radioOptions
                        ]
                    if visibilityRules is not None:
                        existing_control.conditionalVisibility = visibilityRules
                    return {
                        "status": "success",
                        "message": f"Updated control '{existing_control.name}'",
                    }
            control_config = {
                "name": new_name,
                "label": new_label,
                "visibleLabel": True,
                "type_": controlType,
                "visible": visible,
                "validators": (
                    [IValidator(required=required, **(validation or {}))]
                    if required or validation
                    else None
                ),
                "dependentControls": dependentControls,
                "conditionalVisibility": visibilityRules,
            }
            if radioOptions:
                processed_options = []
                for opt in radioOptions:
                    if not all(key in opt for key in ["name", "label", "value"]):
                        raise ValueError(
                            f"Invalid radio option: {opt}. Must have name, label, and value."
                        )
                    radio_opt = {
                        "name": opt["name"],
                        "label": opt["label"],
                        "value": opt["value"],
                        "selected": opt.get("selected", False),
                        "visible": opt.get("visible", True),
                        "dependentControls": (
                            [
                                {"name": dc["name"], "visibility": dc["visibility"]}
                                for dc in opt.get("dependentControls", [])
                            ]
                            if "dependentControls" in opt
                            else []
                        ),
                    }
                    processed_options.append(radio_opt)
                control_config["radioOptions"] = [
                    IRadioOption(**opt) for opt in processed_options
                ]
            control = IFormControl(**control_config)
            target_section.formControls.append(control)
            if dependentControls:
                self._add_dependent_controls(target_section, control, dependentControls)
            return {
                "status": "success",
                "message": f"Added new control '{new_name}' to section '{normalized_title}'",
                "control_name": new_name,
                "section": normalized_title,
                "dependent_controls": [d["name"] for d in (dependentControls or [])],
            }
        except Exception as e:
            return {
                "error": f"Failed to add/update control: {str(e)}",
                "section": normalized_title,
                "control_name": new_name if "new_name" in locals() else None,
            }

    def _add_dependent_controls(
        self,
        section: IFormSections,
        parent_control: IFormControl,
        dependent_controls: List[Dict[str, Any]],
    ) -> None:
        """Add dependent controls to a section."""
        for dep in dependent_controls:
            if not all(key in dep for key in ["name", "label", "type"]):
                raise ValueError(
                    "Each dependent control must have 'name', 'label', and 'type' fields."
                )
            dep_config = {
                "name": dep["name"],
                "label": dep["label"],
                "type_": dep["type"],
                "visibleLabel": True,
                "visible": False,
                "conditionalVisibility": {
                    "dependsOn": parent_control.name,
                    "values": dep.get("showOnValues", ["Y"]),
                },
                "validators": (
                    [IValidator(**v) for v in dep.get("validators", [])]
                    if dep.get("validators")
                    else None
                ),
            }
            if dep.get("required"):
                if not dep_config["validators"]:
                    dep_config["validators"] = []
                dep_config["validators"].append(
                    IValidator(required=True, message=f"{dep['label']} is required")
                )
            dep_control = IFormControl(**dep_config)
            section.formControls.append(dep_control)

    def _update_dependent_controls(
        self,
        sectionTitle: str,
        controlName: str,
        radioOptions: List[Dict[str, Any]],
        delete_dependent: bool = False,
    ) -> Dict:
        """Update or delete dependent controls."""
        try:
            target_section = next(
                (
                    section
                    for section in self.form.formSections
                    if section.sectionTitle == sectionTitle
                ),
                None,
            )
            if not target_section:
                return {"error": f"Section '{sectionTitle}' not found"}
            target_control = next(
                (
                    control
                    for control in target_section.formControls
                    if control.name == controlName
                    or control.label.lower() == controlName.lower()
                ),
                None,
            )
            if not target_control:
                return {"error": f"Control '{controlName}' not found"}
            if delete_dependent:
                dependent_names = set()
                for option in radioOptions:
                    if "dependentControls" in option:
                        for dep in option["dependentControls"]:
                            dependent_names.add(dep["name"])
                for dep_name in dependent_names:
                    self.delete_control(sectionTitle, dep_name, is_dependent=True)
                return {
                    "status": "success",
                    "message": f"Deleted {len(dependent_names)} dependent controls",
                }
            if not hasattr(target_control, "radioOptions"):
                target_control.radioOptions = []
            for option in radioOptions:
                if not all(key in option for key in ["name", "dependentControls"]):
                    return {"error": f"Invalid radio option format: {option}"}
                radio_option = next(
                    (
                        opt
                        for opt in target_control.radioOptions
                        if opt.name == option["name"]
                    ),
                    None,
                )
                if radio_option:
                    radio_option.dependentControls = option["dependentControls"]
                else:
                    target_control.radioOptions.append(IRadioOption(**option))
            return {
                "status": "success",
                "message": f"Updated dependent controls for {controlName}",
                "updated_options": [opt.name for opt in target_control.radioOptions],
            }
        except Exception as e:
            return {"error": f"Error updating dependent controls: {str(e)}"}

    def get_dependent_controls(self, sectionTitle: str, controlName: str) -> Dict:
        """Get dependent controls for a specific control."""
        for section in self.form.formSections:
            if section.sectionTitle == sectionTitle:
                for control in section.formControls:
                    if (
                        control.name == controlName
                        or control.label.lower() == controlName.lower()
                    ):
                        return {
                            "status": "success",
                            "dependent_controls": control.dependentControls or [],
                        }
                return {
                    "error": f"Control '{controlName}' not found in section '{sectionTitle}'"
                }
        return {"error": f"Section '{sectionTitle}' not found"}

    def delete_section(self, sectionTitle: str) -> Dict:
        """Delete a section from the form."""
        for i, section in enumerate(self.form.formSections):
            if section.sectionTitle == sectionTitle:
                self.form.formSections.pop(i)
                return {
                    "status": "success",
                    "message": f"Section '{sectionTitle}' deleted",
                }
        return {"error": f"Section '{sectionTitle}' not found"}

    def delete_control(
        self, sectionTitle: str, controlName: str, is_dependent: bool = False
    ) -> Dict:
        """Delete a control and all references to it from the form."""
        try:
            target_section = next(
                (
                    section
                    for section in self.form.formSections
                    if section.sectionTitle == sectionTitle
                ),
                None,
            )
            if not target_section:
                return {"error": f"Section '{sectionTitle}' not found"}
            control_deleted = False
            controls_to_keep = []
            for control in target_section.formControls:
                if (
                    control.name == controlName
                    or control.label.lower() == controlName.lower()
                ):
                    control_deleted = True
                    continue
                controls_to_keep.append(control)
            if control_deleted:
                target_section.formControls = controls_to_keep
                for section in self.form.formSections:
                    for control in section.formControls:
                        if hasattr(control, "radioOptions"):
                            for option in control.radioOptions:
                                if hasattr(option, "dependentControls"):
                                    option.dependentControls = [
                                        dep
                                        for dep in option.dependentControls
                                        if dep["name"] != controlName
                                    ]
                        if hasattr(control, "options"):
                            for option in control.options:
                                if hasattr(option, "dependentControls"):
                                    option.dependentControls = [
                                        dep
                                        for dep in option.dependentControls
                                        if dep["name"] != controlName
                                    ]
                        if hasattr(control, "dependentControls"):
                            control.dependentControls = [
                                dep
                                for dep in control.dependentControls
                                if dep["name"] != controlName
                            ]
                return {
                    "status": "success",
                    "message": f"Control '{controlName}' and all references deleted",
                }
            return {
                "error": f"Control '{controlName}' not found in section '{sectionTitle}'"
            }
        except Exception as e:
            return {"error": f"Error deleting control: {str(e)}"}
        
    def update_control_validation(
        self, sectionTitle: str, controlName: str, validation: Dict
    ) -> Dict:
        """
        Update or add validation rules for a control.
        Handles any validation field defined in IValidator.
        """
        target_section = next(
            (
                section
                for section in self.form.formSections
                if section.sectionTitle == sectionTitle
            ),
            None,
        )
        if not target_section:
            return {"error": f"Section '{sectionTitle}' not found"}
        target_control = next(
            (
                control
                for control in target_section.formControls
                if control.name == controlName
                or control.label.lower() == controlName.lower()
            ),
            None,
        )
        if not target_control:
            return {
                "error": f"Control '{controlName}' not found in section '{sectionTitle}'"
            }
        if not target_control.validators:
            target_control.validators = []
        for key, value in validation.items():
            normalized_key = key[0].lower() + key[1:]
            if normalized_key == "minlength":
                normalized_key = "minLength"
            elif normalized_key == "maxlength":
                normalized_key = "maxLength"
            existing_validator = next(
                (
                    v
                    for v in target_control.validators
                    if getattr(v, "validatorName", None) == normalized_key
                ),
                None,
            )
            if existing_validator:
                setattr(existing_validator, normalized_key, value)
            else:
                validator_data = {"validatorName": normalized_key}
                if normalized_key == "required":
                    validator_data["required"] = True
                    validator_data["message"] = validation.get(
                        "message", "This field is required."
                    )
                elif normalized_key == "pattern":
                    validator_data["pattern"] = value
                    validator_data["message"] = validation.get(
                        "message", "Invalid format."
                    )
                elif normalized_key == "minLength":
                    validator_data["minLength"] = value
                    validator_data["message"] = validation.get(
                        "message", f"Minimum length is {value}."
                    )
                elif normalized_key == "maxLength":
                    validator_data["maxLength"] = value
                    validator_data["message"] = validation.get(
                        "message", f"Maximum length is {value}."
                    )
                else:
                    return {"error": f"Unsupported validator: {normalized_key}"}
                target_control.validators.append(IValidator(**validator_data))
        return {
            "status": "success",
            "message": f"Validation updated for control '{target_control.name}'",
        }

    def set_form_title(self, title: str) -> Dict:
        """Set the form title."""
        self.form.formTitle = title
        return {"status": "success", "form_title": title}

    def is_form_complete(self) -> bool:
        """Check if the form has all expected controls."""
        total_controls = sum(
            len(section.formControls) for section in self.form.formSections
        )
        has_sufficient_structure = len(self.form.formSections) > 0 and any(
            len(section.formControls) > 0 for section in self.form.formSections
        )
        min_expected = max(1, self.expected_controls_count)
        return has_sufficient_structure and total_controls >= min_expected

    def remove_duplicate_controls(self) -> Dict:
        """Remove duplicate controls that have the same label within the same section."""
        duplicates_removed = 0
        for section in self.form.formSections:
            seen_labels = {}
            controls_to_keep = []
            for control in section.formControls:
                label_key = control.label.lower()
                if label_key not in seen_labels:
                    seen_labels[label_key] = control
                    controls_to_keep.append(control)
                else:
                    duplicates_removed += 1
            section.formControls = controls_to_keep
        return {
            "status": "success",
            "message": f"Removed {duplicates_removed} duplicate controls",
            "duplicates_removed": duplicates_removed,
        }

    def get_current_form(self) -> Dict:
        """
        Return the current form state with null values removed and 'type_' renamed to 'type'.
        """
        try:
            # Get the raw form data
            form_data = self.form.model_dump()
            
            # Clean null values
            cleaned_data = clean_null_values(form_data)
            
            # Transform 'type_' fields back to 'type'
            self.transform_type_back(cleaned_data)
            
            return cleaned_data
        except AttributeError:
            # Fallback for older Pydantic versions or custom models
            form_data = self.form.dict()
            
            # Clean null values
            cleaned_data = clean_null_values(form_data)
            
            # Transform 'type_' fields back to 'type'
            self.transform_type_back(cleaned_data)
            
            return cleaned_data
    
        
    def transform_type_back(self, data: Any) -> None:
        """
        Recursively transform 'type_' to 'type' and 'class_' to 'class' in the given data structure.
        """
        if isinstance(data, dict):
            # Handle type_ conversion
            if 'type_' in data:
                data['type'] = data.pop('type_')
                
            # Handle class_ conversion    
            if 'class_' in data:
                data['class'] = data.pop('class_')
                
            # Recursively process all values
            for key, value in data.items():
                self.transform_type_back(value)
                
        elif isinstance(data, list):
            # Recursively process each item in the list
            for item in data:
                self.transform_type_back(item)


    def load_form_data(self, form_data: Dict) -> Dict:
        """
        Load existing form data into the form builder.
        - Transforms the JSON data to match Pydantic models.
        - Processes sections, controls, dependent controls, and visibility rules.
        """
        try:
            _transform_dependent_controls(form_data)
            transformed_data = transform_json_for_pydantic(form_data)
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
                "saveBtnFunction": None,
            }
            for key, value in transformed_data.items():
                if key == "formSections":
                    processed_sections = []
                    for section in value:
                        section_data = {
                            "sectionTitle": section.get(
                                "sectionTitle", "Untitled Section"
                            ),
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
                            "urlDependentControls": section.get(
                                "urlDependentControls", None
                            ),
                            "urlPath": section.get("urlPath", None),
                            "productFeaturesUrl": section.get(
                                "productFeaturesUrl", None
                            ),
                        }
                        if "formControls" in section and isinstance(
                            section["formControls"], list
                        ):
                            for control in section["formControls"]:
                                dependent_controls = control.get(
                                    "dependentControls", []
                                )
                                if dependent_controls:
                                    for dep_control in dependent_controls:
                                        if not all(
                                            key in dep_control
                                            for key in ["name", "visibility"]
                                        ):
                                            raise ValueError(
                                                f"Invalid dependent control: {dep_control}. Each dependent control must have 'name' and 'visibility' fields."
                                            )
                                control_data = {
                                    "name": control.get(
                                        "name",
                                        f"control_{len(section_data['formControls'])}",
                                    ),
                                    "label": control.get("label", "Untitled Control"),
                                    "visibleLabel": control.get("visibleLabel", True),
                                    "type_": control.get("type_", "text"),
                                    "validators": control.get("validators", None),
                                    "dependentControls": dependent_controls,
                                    "conditionalVisibility": control.get(
                                        "conditionalVisibility", None
                                    ),
                                    "radioOptions": (
                                        [
                                            IRadioOption(**opt)
                                            for opt in control.get("radioOptions", [])
                                        ]
                                        if "radioOptions" in control
                                        else None
                                    ),
                                }
                                section_data["formControls"].append(
                                    IFormControl(**control_data)
                                )
                        processed_sections.append(IFormSections(**section_data))
                    processed_data["formSections"] = processed_sections
                else:
                    processed_data[key] = value
            self.form = IForm(**processed_data)
            total_controls = sum(
                len(section.formControls) for section in self.form.formSections
            )
            self.expected_controls_count = total_controls
            self.current_form_id = processed_data.get("id", None)
            return {
                "status": "success",
                "message": f"Form loaded successfully with {total_controls} controls",
            }
        except Exception as e:
            print(f"Exception in load_form_data: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": f"Error loading form data: {str(e)}"}

builder = FormBuilder()

def get_form_templates_list() -> List[str]:
    """Get a list of available form template filenames."""
    templates_dir = Path("src/form_builder/form_templates")
    if not templates_dir.exists():
        templates_dir = Path("form_templates")
    templates = []
    if templates_dir.exists():
        templates = [f.name for f in templates_dir.glob("*.json")]
        templates.extend([f.name for f in templates_dir.glob("*.txt")])
    return templates

def load_form_template(template_name: str) -> Dict:
    """Load a form template from the form_templates directory."""
    templates_dir = Path("src/form_builder/form_templates")
    if not templates_dir.exists():
        templates_dir = Path("form_templates")
    if not template_name.endswith((".json", ".txt")):
        json_path = templates_dir / f"{template_name}.json"
        txt_path = templates_dir / f"{template_name}.txt"
        if json_path.exists():
            template_path = json_path
        elif txt_path.exists():
            template_path = txt_path
        else:
            raise FileNotFoundError(
                f"Template '{template_name}' not found with either .json or .txt extension"
            )
    else:
        template_path = templates_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template '{template_name}' not found")
    with open(template_path, "r") as f:
        content = f.read()
        if template_path.suffix.lower() == ".txt":
            try:
                template_data = json.loads(content)
            except json.JSONDecodeError:
                raise ValueError(
                    f"The .txt file '{template_name}' does not contain valid JSON data"
                )
        else:
            template_data = json.loads(content)
    return template_data

tools = {
    "add_section": {
        "description": "Adds a new section to the form.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {
                    "type": "string",
                    "description": "Title of the section",
                }
            },
            "required": ["sectionTitle"],
        },
        "func": builder.add_section,
    },
    "add_control": {
        "description": "Adds a new control to a section with support for dependent controls and radio options.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {
                    "type": "string",
                    "description": "Title of the section where the control will be added.",
                },
                "controlType": {
                    "type": "string",
                    "description": "Type of the control (e.g., text, email, number, date, select, textarea, checkbox, radio).",
                },
                "label": {
                    "type": "string",
                    "description": "Label for the control. This will be displayed to the user.",
                },
                "name": {
                    "type": "string",
                    "description": "Unique name identifier for the control. If not provided, it will be generated from the label.",
                },
                "required": {
                    "type": "boolean",
                    "description": "Whether the control is required. Defaults to false if not specified.",
                },
                "validation": {
                    "type": "object",
                    "description": "Validation rules for the control (e.g., minLength, maxLength, pattern).",
                    "additionalProperties": True,
                },
                "radioOptions": {
                    "type": "array",
                    "description": "Options for radio button controls. Each option can specify dependent controls.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Unique name for the radio option.",
                            },
                            "label": {
                                "type": "string",
                                "description": "Label for the radio option. This will be displayed to the user.",
                            },
                            "value": {
                                "type": "string",
                                "description": "Value associated with the radio option.",
                            },
                            "selected": {
                                "type": "boolean",
                                "description": "Whether this option is selected by default.",
                            },
                            "visible": {
                                "type": "boolean",
                                "description": "Whether this option is visible by default.",
                            },
                            "dependentControls": {
                                "type": "array",
                                "description": "List of dependent controls that should be shown or hidden based on this option.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Name of the dependent control.",
                                        },
                                        "visibility": {
                                            "type": "boolean",
                                            "description": "Whether the dependent control should be visible when this option is selected.",
                                        },
                                    },
                                    "required": ["name", "visibility"],
                                },
                            },
                        },
                        "required": ["name", "label", "value"],
                    },
                },
                "dependentControls": {
                    "type": "array",
                    "description": "List of controls that depend on this control's value for visibility.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Name of the dependent control.",
                            },
                            "label": {
                                "type": "string",
                                "description": "Label of the dependent control.",
                            },
                            "type": {
                                "type": "string",
                                "description": "Type of the dependent control (e.g., text, date).",
                            },
                            "visible": {
                                "type": "boolean",
                                "description": "Initial visibility state of the dependent control.",
                            },
                        },
                        "required": ["name", "type", "visible"],
                    },
                },
                "visibilityRules": {
                    "type": "object",
                    "description": "Rules for conditional visibility of the control based on other controls' values.",
                    "additionalProperties": True,
                },
            },
            "required": ["sectionTitle", "controlType", "label"],
        },
        "func": builder.add_control,
    },
    "delete_section": {
        "description": "Deletes a section from the form.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {
                    "type": "string",
                    "description": "Title of the section to delete",
                }
            },
            "required": ["sectionTitle"],
        },
        "func": builder.delete_section,
    },
    "delete_control": {
        "description": "Deletes a control from a section and remove all references to it.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {
                    "type": "string",
                    "description": "Title of the section",
                },
                "controlName": {
                    "type": "string",
                    "description": "Name or label of the control to delete",
                },
                "is_dependent": {
                    "type": "boolean",
                    "description": "Whether this is a dependent control deletion",
                },
            },
            "required": ["sectionTitle", "controlName"],
        },
        "func": builder.delete_control,
    },
    "update_control_validation": {
        "description": "Updates validation rules for a control.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {
                    "type": "string",
                    "description": "Title of the section",
                },
                "controlName": {
                    "type": "string",
                    "description": "Name or label of the control",
                },
                "validation": {
                    "type": "object",
                    "description": "Updated validation rules",
                },
            },
            "required": ["sectionTitle", "controlName", "validation"],
        },
        "func": builder.update_control_validation,
    },
    "set_form_title": {
        "description": "Sets the title of the form.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title for the form"}
            },
            "required": ["title"],
        },
        "func": builder.set_form_title,
    },
    "_update_dependent_controls": {
        "description": "Updates dependent control relationships for radio button options.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {"type": "string"},
                "controlName": {"type": "string"},
                "radioOptions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "dependentControls": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "visibility": {"type": "boolean"},
                                    },
                                },
                            },
                        },
                    },
                },
                "delete_dependent": {"type": "boolean"},
            },
            "required": ["sectionTitle", "controlName", "radioOptions"],
        },
        "func": builder._update_dependent_controls,
    },
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

    field_pattern = r"([\w\s]+)\s*\(([\w\s,]+)\)"
    field_matches = re.findall(field_pattern, prompt, re.IGNORECASE)
    total_controls = len(field_matches)
    builder.set_expected_controls(total_controls)
    section_idx = 0
    fields_by_section = {section: [] for section in sections}
    for field_name, field_details in field_matches:
        fields_by_section[sections[section_idx]].append((field_name, field_details))
        section_idx = (section_idx + 1) % len(sections)
    for section, fields in fields_by_section.items():
        fields_with_validations = []
        for field_name, field_details in fields:
            field_name = field_name.strip()
            validations = {}
            field_type = None
            

            # In the field_pattern section, update the type detection logic:
        if "text" in field_details.lower():
            field_type = "text"
        elif "email" in field_details.lower():
            field_type = "email"
        elif "phone" in field_details.lower() or "mobile" in field_details.lower():
            field_type = "phonenumber"  # Added phone number type
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
        elif "paragraph" in field_details.lower():
            field_type = "paragraph"
        elif "line" in field_details.lower():
            field_type = "line"
        elif "fileupload" in field_details.lower() or "file upload" in field_details.lower():
            field_type = "fileupload"
        elif "button" in field_details.lower():
            field_type = "button"
        elif "multiselect" in field_details.lower():
            field_type = "multiselect"
        elif "bold" in field_details.lower():
            field_type = "bold"
        elif "summary" in field_details.lower():
            field_type = "summary"
        elif "password" in field_details.lower():
            field_type = "password"
        else:
            field_type = "text"
              # Default to text if no type specified
        

        field_config = {
            "field": field_name,
            "type": field_type,
            "validations": {}
        }

# Add phone number specific validation if needed
        if field_type == "phonenumber":
            field_config["validations"] = {
                "validators": [
                    {
                        "validatorName": "required",
                        "required": True,
                        "message": "Phone number is required"
                    },
                    {
                        "validatorName": "pattern",
                        "pattern": "^[6-9]\\d{9}$",  # Indian mobile number format
                        "message": "Please enter a valid 10-digit phone number"
                    }
                ]
            }

            for key, keyword_pattern in field_validation_keywords.items():
                validation_match = re.search(
                    keyword_pattern, field_details, re.IGNORECASE
                )
                if validation_match:
                    if key in ["max_length", "min_length", "min_value", "max_value"]:
                        validations[key] = int(validation_match.group(2))
                    elif key in ["required", "email"]:
                        validations[key] = True
                    elif key == "pattern":
                        validations[key] = validation_match.group(2).strip()
            fields_with_validations.append(
                {"field": field_name, "type": field_type, "validations": validations}
            )
        if fields_with_validations:
            extracted_data.append(
                {"section": section, "controls": fields_with_validations}
            )
    return extracted_data

async def detect_request_type(prompt: str) -> bool:
    """Use Gemini to determine if the user wants to modify an existing form or create a new one."""
    current_form = builder.get_current_form()
    has_existing_form = len(current_form.get("formSections", [])) > 0 and any(
        len(section.get("formControls", [])) > 0
        for section in current_form.get("formSections", [])
    )
    if not has_existing_form:
        return False
    if builder.current_form_id is not None:
        return True
    modification_keywords = [
        "add",
        "change",
        "modify",
        "update",
        "delete",
        "remove",
        "edit",
        "alter",
        "adjust",
        "revise",
        "include",
        "append",
    ]
    for keyword in modification_keywords:
        if re.search(r"\b" + keyword + r"\b", prompt, re.IGNORECASE):
            return True
    try:
        system_prompt = """
        You are a request classifier for a form builder application. 
        Your task is to determine whether the user's request is: 
        1. Creating a NEW form from scratch, or
        2. MODIFYING an existing form
        Examine the request and respond with ONLY "NEW" or "MODIFY".
        """
        chat = model.start_chat()
        response = chat.send_message(
            f"{system_prompt}\n\nUser request: {prompt}",
            generation_config={"temperature": 0.1},
        )
        result = response.text.strip().upper()
        is_modification = "MODIFY" in result
        print(f"Request classification: {'MODIFY' if is_modification else 'NEW'}")
        return is_modification
    except Exception as e:
        print(f"Error classifying request: {e}")
        return False

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
    is_json_modification: bool
    successful_tools: int
    failed_tools: int

workflow = Graph()

def _clean_control_references(self, controlName: str) -> None:
    """Remove all references to a control throughout the form."""
    for section in self.form.formSections:
        for control in section.formControls:
            if hasattr(control, "radioOptions"):
                for option in control.radioOptions:
                    if hasattr(option, "dependentControls"):
                        option.dependentControls = [
                            dep
                            for dep in option.dependentControls
                            if dep["name"] != controlName
                        ]
            if hasattr(control, "options"):
                for option in control.options:
                    if hasattr(option, "dependentControls"):
                        option.dependentControls = [
                            dep
                            for dep in option.dependentControls
                            if dep["name"] != controlName
                        ]
            if hasattr(control, "dependentControls"):
                control.dependentControls = [
                    dep
                    for dep in control.dependentControls
                    if dep["name"] != controlName
                ]

@workflow.add_node
async def start_node(state: WorkflowState) -> WorkflowState:
    """Process the initial user input."""
    prompt = state["prompt"]
    is_json_modification = state.get("is_json_modification", False)
    if not is_json_modification:
        is_modification = await detect_request_type(prompt)
        if not is_modification:
            builder.reset_form()
            extracted_data = extract_sections_and_controls(prompt)
        else:
            extracted_data = []
    else:
        is_modification = True
        extracted_data = []
    system_prompt = ""
    if is_modification:
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
                            if key != "type_":
                                validation_details.append(f"{key}: {value}")
                    if validation_details:
                        validation_info = (
                            f" with validation ({', '.join(validation_details)})"
                        )
                controls_info.append(
                    f"{control.get('name')}: {control.get('label')} ({control.get('type')}){validation_info}"
                )
            if controls_info:
                section_detail = f"Section {idx+1}: {section.get('sectionTitle')}\n"
                section_detail += "\n".join(
                    [f"  - {control}" for control in controls_info]
                )
                sections_info.append(section_detail)
            else:
                section_detail = (
                    f"Section {idx+1}: {section.get('sectionTitle')} (empty)"
                )
                sections_info.append(section_detail)
        form_details = "\n\n".join(sections_info)
        source_info = (
            "from JSON input" if is_json_modification else "from existing session"
        )
        fields_to_add = []
        field_pattern = r"([\w\s]+)\s*\(([\w\s,]+)\)"
        field_matches = re.findall(field_pattern, prompt, re.IGNORECASE)
        if field_matches:
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
        7. _update_dependent_controls - Update the dependent controls of an existing control
        When modifying controls:
        - For delete_control and update_control_validation, you can use either the control's name or label.
        - For add_control, ensure the label doesn't already exist in that section.
        - Make sure to specify all required parameters for each tool call.
        - Be precise with section titles and control names/labels.
        When using add_control:
        - If the control already exists (by name or label):
        - Only the requested field (label or name) will be updated, and the other field will remain unchanged.
        - If the control does not exist, a new control will be added with the name set to the camelCase version of the label.
        When adding a radio button control:
        - Include 'dependentControls' for each option to specify which controls should be shown or hidden.
        - Each option must define 'dependentControls' as a list of objects with 'name' and 'visibility' fields.
        - Example:
        [
            {{
                "name": "add_control",
                "parameters": {{
                    "sectionTitle": "Policy Details",
                    "controlType": "radio",
                    "label": "Do you have an existing policy?",
                    "name": "hasExistingPolicy",
                    "radioOptions": [
                        {{
                            "name": "yes",
                            "label": "Yes",
                            "value": "Y",
                            "dependentControls": [
                                {{"name": "policyNumber", "visibility": true}},
                                {{"name": "policyStartDate", "visibility": true}}
                            ]
                        }},
                        {{
                            "name": "no",
                            "label": "No",
                            "value": "N",
                            "dependentControls": [
                                {{"name": "policyNumber", "visibility": false}},
                                {{"name": "policyStartDate", "visibility": false}}
                            ]
                        }}
                    ]
                }}
            }}
        ]
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
            {"role": "user", "content": prompt},
        ],
        "iteration_count": 0,
        "extracted_data": extracted_data,
        "is_modification": is_modification,
        "is_json_modification": is_json_modification,
        "successful_tools": 0,
        "failed_tools": 0,
    }


def extract_tool_calls_from_message(content: str) -> List[Dict]:
    """Extract tool calls from LLM messages even when formatted poorly."""
    tool_calls = []
    try:
        cleaned_content = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", content)
        cleaned_content = cleaned_content.strip()
        if not (cleaned_content.startswith("[") and cleaned_content.endswith("]")):
            json_pattern = r"\[\s*{[\s\S]*}\s*\]"
            json_match = re.search(json_pattern, cleaned_content)
            if json_match:
                cleaned_content = json_match.group(0)
        parsed_content = json.loads(cleaned_content)
        if isinstance(parsed_content, list):
            return [
                item
                for item in parsed_content
                if isinstance(item, dict) and "name" in item and "parameters" in item
            ]
        elif isinstance(parsed_content, dict):
            if "name" in parsed_content and "parameters" in parsed_content:
                return [parsed_content]
    except Exception as e:
        print(f"Initial JSON parsing failed: {e}")
    try:
        normalized_content = re.sub(r"([{,])\s*([a-zA-Z_]+):", r'\1"\2":', content)
        tool_pattern = (
            r'{"name":\s*"([^"]+)",\s*"parameters":\s*({[^{}]*(?:{[^{}]*}[^{}]*)*})'
        )
        matches = re.findall(tool_pattern, normalized_content)
        for name, params_str in matches:
            try:
                params_str = re.sub(r"([{,])\s*([a-zA-Z_]+):", r'\1"\2":', params_str)
                params = json.loads(params_str)
                tool_calls.append({"name": name, "parameters": params})
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
    if iteration_count >= 20:
        return {
            "messages": messages,
            "iteration_count": iteration_count,
            "tool_calls": [],
            "form_complete": True,
            "is_modification": is_modification,
            "successful_tools": successful_tools,
            "failed_tools": failed_tools,
        }
    try:
        gemini_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                gemini_messages.append(
                    {
                        "role": "user",
                        "parts": [{"text": "[System Instructions]\n" + msg["content"]}],
                    }
                )
            elif role == "user":
                gemini_messages.append(
                    {"role": "user", "parts": [{"text": msg["content"]}]}
                )
            elif role == "assistant":
                gemini_messages.append(
                    {"role": "model", "parts": [{"text": msg["content"]}]}
                )
            elif role == "tool":
                gemini_messages.append(
                    {
                        "role": "user",
                        "parts": [{"text": "[Tool Results]\n" + msg["content"]}],
                    }
                )
        chat = model.start_chat(history=gemini_messages)
        response = chat.send_message(
            messages[-1]["content"], generation_config={"temperature": 0.5}
        )
        content = response.text
        print(f"Gemini Response: {content}")
        tool_calls = extract_tool_calls_from_message(content)
        print(f"Initial Parsed Tool Calls: {tool_calls}")
        valid_tool_calls = []
        seen_operations = set()
        for call in tool_calls:
            if isinstance(call, dict) and "name" in call and "parameters" in call:
                if call["name"] in tools:
                    if call["name"] == "_update_dependent_controls":
                        section = call["parameters"].get("sectionTitle", "")
                        control = call["parameters"].get("controlName", "")
                        op_signature = f"update_dependent:{section}:{control}"
                    if call["name"] == "add_control":
                        section = call["parameters"].get("sectionTitle", "")
                        label = call["parameters"].get("label", "")
                        op_signature = f"add_control:{section}:{label}"
                    elif call["name"] in [
                        "delete_control",
                        "update_control_validation",
                    ]:
                        section = call["parameters"].get("sectionTitle", "")
                        control = call["parameters"].get("controlName", "")
                        op_signature = f"{call['name']}:{section}:{control}"
                    else:
                        op_signature = (
                            f"{call['name']}:{json.dumps(call['parameters'])}"
                        )
                    if op_signature not in seen_operations:
                        valid_tool_calls.append(call)
                        seen_operations.add(op_signature)
        tool_calls = valid_tool_calls
        print(f"Validated Tool Calls: {tool_calls}")
        if is_modification:
            form_complete = (
                len(valid_tool_calls) == 0 and iteration_count > 1
            ) or iteration_count >= 5
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
            "failed_tools": failed_tools,
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
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tools:
            try:
                print(
                    f"Executing tool '{tool_name}' with parameters: {tool_call['parameters']}"
                )
                result = tools[tool_name]["func"](**tool_call["parameters"])
                print(f"Executed tool '{tool_name}' with result: {result}")
                if "status" in result and result["status"] in ["success", "warning"]:
                    successful_tools += 1
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (IFormSections, IFormControl)):
                            result[key] = value.dict()
                results.append(
                    {
                        "tool_call_id": tool_call.get("id", ""),
                        "name": tool_name,
                        "result": result,
                    }
                )
            except Exception as e:
                print(f"Error executing tool '{tool_name}': {e}")
                failed_tools += 1
                error_message = str(e)
                if "sectionTitle" in str(e).lower():
                    error_message = f"Missing or invalid sectionTitle parameter. Available sections: {[s.sectionTitle for s in builder.form.formSections]}"
                elif "controlName" in str(e).lower():
                    error_message = f"Missing or invalid controlName parameter. For this tool, you can use either the control's name (e.g., 'control_0') or its label."
                results.append(
                    {
                        "tool_call_id": tool_call.get("id", ""),
                        "name": tool_name,
                        "error": error_message,
                    }
                )
    total_controls = sum(
        len(section.formControls) for section in builder.form.formSections
    )
    expected_controls = builder.expected_controls_count
    print(f"Progress: {total_controls}/{expected_controls} controls added")
    if is_modification:
        form_complete = (len(tool_calls) == 0 or iteration_count >= 5) and (
            successful_tools > 0 or iteration_count >= 5
        )
    else:
        form_complete = builder.is_form_complete() and (
            successful_tools > 0 or iteration_count >= 5
        )
    print(
        f"Form complete: {form_complete} (Success: {successful_tools}, Failed: {failed_tools})"
    )
    updated_form = builder.get_current_form()
    if failed_tools > 0 and successful_tools == 0:
        available_sections = [s.sectionTitle for s in builder.form.formSections]
        controls_by_section = {}
        for section in builder.form.formSections:
            controls_by_section[section.sectionTitle] = [
                f"{c.name} (label: '{c.label}', type: {c.type_})"
                for c in section.formControls
            ]
        context_message = {
            "role": "user",
            "content": f"""
                All tool calls failed. Please correct your approach:
                1. Available sections: {
                            available_sections}
                2. Controls in each section: {json.dumps(controls_by_section, indent=2)}
                3. For delete_control and update_control_validation, you can use either the control's name or label
                4. For add_control, ensure the label doesn't already exist in that section
                5. For update_control_validation, provide complete validation rules
                Try again with EXACT section names and proper parameters.
                """,}
        messages.append(context_message)
    return {
        "messages": messages
        + [
            {
                "role": "tool",
                "content": json.dumps(results),
                "tool_call_id": tool_call.get("id", "") if tool_calls else "",
            }
        ],
        "results": results,
        "form_complete": form_complete,
        "is_modification": is_modification,
        "iteration_count": iteration_count,
        "updated_form": updated_form,
        "successful_tools": successful_tools,
        "failed_tools": failed_tools,
    }

@workflow.add_node
def end_node(state: WorkflowState) -> WorkflowState:
    """Finalize the form generation and clean up any duplicates."""
    duplicate_result = builder.remove_duplicate_controls()
    print(f"Duplication result: {duplicate_result}")
    current_form = builder.get_current_form()
    return {
        "form": current_form,
        "updated_form": current_form,
        "is_modification": state.get("is_modification", False),
        "successful_tools": state.get("successful_tools", 0),
        "failed_tools": state.get("failed_tools", 0),
        "deduplication": duplicate_result,
    }
    
workflow.set_entry_point("start_node")
workflow.add_edge("start_node", "llm_node")

workflow.add_conditional_edges(
    "llm_node",
    lambda state: (
        state.get("form_complete", False)
        or (not state.get("tool_calls") and not state.get("retry_parsing", False))
    ),
    {True: "end_node", False: "tool_node"},
)

workflow.add_conditional_edges(
    "tool_node",
    lambda state: state.get("form_complete", False),
    {True: "end_node", False: "llm_node"},
)

workflow.add_edge("end_node", END)

app_workflow = workflow.compile()


@app.post("/load-json-form")
async def load_json_form(form_data: Dict) -> Dict:
    """Load a form from JSON data."""
    try:
        result = builder.load_form_data(form_data)
        if result.get("status") == "error":
            raise HTTPException(
                status_code=400, detail=result.get("message", "Error loading form data")
            )
        return {
            "status": "success",
            "result": result,
            "form": builder.get_current_form(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FormRequestModel(BaseModel):
    prompt: str

class FormModificationRequest(BaseModel):
    prompt: str
    form_json: Optional[Dict] = None

@app.post("/update-dependent-controls")
async def update_dependent_controls(
    section_title: str,
    control_name: str,
    radio_options: List[Dict[str, Any]],
    delete_dependent: bool = False,
) -> Dict:
    """Update or delete dependent controls for a specific control."""
    try:
        for option in radio_options:
            if "name" not in option or "dependentControls" not in option:
                raise ValueError(
                    "Each radio option must have 'name' and 'dependentControls'"
                )
            for dep in option["dependentControls"]:
                if "name" not in dep:
                    raise ValueError("Each dependent control must have 'name'")
                if not delete_dependent and "visibility" not in dep:
                    raise ValueError(
                        "Each dependent control must have 'visibility' when updating"
                    )
        result = builder._update_dependent_controls(
            sectionTitle=section_title,
            controlName=control_name,
            radioOptions=radio_options,
            delete_dependent=delete_dependent,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "status": "success",
            "result": result,
            "form": builder.get_current_form(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating dependent controls: {str(e)}"
        )

@app.get("/get-dependent-controls")
async def get_dependent_controls(section_title: str, control_name: str) -> Dict:
    """Get dependent controls for a specific control."""
    try:
        result = builder.get_dependent_controls(
            sectionTitle=section_title, controlName=control_name
        )
        return {
            "status": "success",
            "result": result,
            "form": builder.get_current_form(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-form")
async def generate_form(
    request: Union[FormRequestModel, FormModificationRequest, TemplateRequestModel],
) -> Dict:
    """Generate or modify a form based on the user's prompt, optional JSON input, or template."""
    try:
        form_json = getattr(request, "form_json", None)
        template_name = getattr(request, "template_name", None)
        if template_name:
            try:
                template_data = load_form_template(template_name)
                load_result = builder.load_form_data(template_data)
                if load_result.get("status") == "error":
                    return {
                        "status": "error",
                        "message": load_result.get("message", "Error loading template"),
                    }
                is_json_modification = True
                print(f"Loaded template '{template_name}' with result: {load_result}")
                print(f"Current form structure: {builder.get_current_form()}")
            except FileNotFoundError:
                return {
                    "status": "error",
                    "message": f"Template '{template_name}' not found",
                }
        elif form_json:
            load_result = builder.load_form_data(form_json)
            if load_result.get("status") == "error":
                return {
                    "status": "error",
                    "message": load_result.get("message", "Error loading form data"),
                }
            is_json_modification = True
            print(f"Loaded form data with result: {load_result}")
            print(f"Current form structure: {builder.get_current_form()}")
        else:
            is_json_modification = False
        config = RunnableConfig(recursion_limit=50)
        result = await app_workflow.ainvoke(
            {"prompt": request.prompt, "is_json_modification": is_json_modification},
            config=config,
        )
        current_form = builder.get_current_form()
        request_type = (
            "template_modification"
            if template_name
            else (
                "json_modification"
                if form_json
                else (
                    "modification"
                    if result.get("is_modification", False)
                    else "new_form"
                )
            )
        )
        return {
            "form": current_form,
            "request_type": request_type,
            "template": template_name if template_name else None,
            "successful_operations": result.get("successful_tools", 0),
            "failed_operations": result.get("failed_tools", 0),
            "deduplication": result.get("deduplication", {}),
        }
    except Exception as e:
        print(f"Error in generate_form: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-section")
async def add_section(title: str) -> Dict:
    """Manually add a section to the form."""
    try:
        result = builder.add_section(sectionTitle=title)
        return {
            "status": "success",
            "result": result,
            "form": builder.get_current_form(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-control")
async def add_control(
    section_title: str,
    control_type: str,
    label: str,
    required: bool = False,
    validation: Optional[Dict] = None,
) -> Dict:
    """Manually add a control to a section."""
    try:
        result = builder.add_control(
            sectionTitle=section_title,
            controlType=control_type,
            label=label,
            required=required,
            validation=validation,
        )
        return {
            "status": "success",
            "result": result,
            "form": builder.get_current_form(),
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
            "form": builder.get_current_form(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-control")
async def delete_control(
    section_title: str, control_name: str, delete_dependents: bool = True
) -> Dict:
    """Delete a control and optionally its dependent controls."""
    try:
        if delete_dependents:
            result = builder._update_dependent_controls(
                sectionTitle=section_title,
                controlName=control_name,
                radioOptions=[],
                delete_dependent=True,
            )
            if "error" in result:
                raise HTTPException(status_code=400, detail=result["error"])
        result = builder.delete_control(
            sectionTitle=section_title, controlName=control_name
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {
            "status": "success",
            "result": result,
            "form": builder.get_current_form(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting control: {str(e)}")

@app.post("/update-validation")
async def update_validation(
    section_title: str, control_name: str, validation: Dict
) -> Dict:
    """Manually update validation rules for a control."""
    try:
        result = builder.update_control_validation(
            sectionTitle=section_title, controlName=control_name, validation=validation
        )
        return {
            "status": "success",
            "result": result,
            "form": builder.get_current_form(),
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
            "form": builder.get_current_form(),
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
            "form": builder.get_current_form(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/form-templates")
async def get_templates() -> Dict:
    """Get list of available form templates."""
    try:
        templates = get_form_templates_list()
        return {"status": "success", "templates": templates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-template")
async def load_template(template_name: str) -> Dict:
    """Load a form template by name."""
    try:
        template_data = load_form_template(template_name)
        result = builder.load_form_data(template_data)
        if result.get("status") == "error":
            raise HTTPException(
                status_code=400, detail=result.get("message", "Error loading template")
            )
        return {
            "status": "success",
            "template": template_name,
            "result": result,
            "form": builder.get_current_form(),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-template")
async def save_template(template_name: str) -> Dict:
    """Save the current form as a modified template."""
    try:
        templates_dir = Path("src/form_builder/form_templates")
        if not templates_dir.exists():
            templates_dir = Path("form_templates")  # Fallback path
            
        if not templates_dir.exists():
            templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Split the filename and extension
        if template_name.endswith(('.json', '.txt')):
            base_name = template_name[:-5] if template_name.endswith('.json') else template_name[:-4]
            extension = '.json' if template_name.endswith('.json') else '.txt'
        else:
            base_name = template_name
            extension = '.json'  # Default to JSON
        
        # Create the modified filename
        modified_name = f"{base_name}_modified{extension}"
        template_path = templates_dir / modified_name
        
        # Get current form data
        form_data = builder.get_current_form()
        
        # Save to file - format based on extension
        with open(template_path, "w") as f:
            if extension == '.json':
                json.dump(form_data, f, indent=2)
            else:  # .txt file
                # Still save as JSON format but in a .txt file
                json_content = json.dumps(form_data, indent=2)
                f.write(json_content)
            
        return {
            "status": "success",
            "message": f"Form saved as modified template '{modified_name}'",
            "template_path": str(template_path),
            "original_template": template_name,
            "modified_template": modified_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=2024)