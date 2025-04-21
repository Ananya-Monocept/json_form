import json
import random
from typing import List, Dict, Any
from form_models import IForm, IFormSections, IFormControl, IValidator
from fastapi import FastAPI
from api import builder, extract_sections_and_controls, tools  # Import from your existing code

# Templates for generating form requests
FORM_TEMPLATES = [
    "Create a {form_type} form with fields for {fields}.",
    "I need a {form_type} form with {fields}.",
    "Build me a {form_type} form that includes {fields}.",
    "Generate a {form_type} form with sections for {sections} and fields including {fields}.",
    "Make a {form_type} form with {fields} and validations such as {validations}.",
]

FORM_TYPES = [
    "contact", "registration", "survey", "feedback", "order", "shipping", 
    "payment", "job application", "event registration", "subscription"
]

SECTIONS = [
    "Personal Information", "Contact Details", "Account Information", 
    "Shipping Address", "Billing Information", "Preferences", 
    "Professional Experience", "Education", "Payment Details"
]

FIELDS = {
    "text": ["Name", "Full Name", "First Name", "Last Name", "Address", "City", "State", "Country", "Zip Code", "Title", "Company", "Position"],
    "email": ["Email", "Email Address", "Alternative Email"],
    "number": ["Age", "Phone Number", "Quantity", "Years of Experience", "Price"],
    "date": ["Birth Date", "Start Date", "End Date", "Registration Date", "Appointment Date"],
    "select": ["Country", "Gender", "Industry", "Job Role", "Subscription Plan"],
    "textarea": ["Comments", "Message", "Additional Information", "Description", "Cover Letter"],
    "checkbox": ["Subscribe to Newsletter", "Accept Terms", "Remember Me", "I agree to the privacy policy"],
    "radio": ["Gender", "Payment Method", "Preferred Contact Method"],
    "file": ["Profile Picture", "Resume", "Supporting Documents"]
}

VALIDATIONS = [
    "required", "email validation", "minimum length of 8", "maximum length of 50",
    "minimum value of 18", "maximum value of 100", "pattern for phone numbers"
]

def generate_random_form_request() -> str:
    """Generate a random form request string."""
    form_type = random.choice(FORM_TYPES)
    
    # Randomly decide if we specify sections
    sections_str = ""
    if random.random() > 0.5:
        sections = random.sample(SECTIONS, k=random.randint(1, 3))
        sections_str = ", ".join(sections)
    
    # Generate random fields
    num_fields = random.randint(3, 8)
    fields = []
    for _ in range(num_fields):
        field_type = random.choice(list(FIELDS.keys()))
        field_name = random.choice(FIELDS[field_type])
        field_desc = f"{field_name} ({field_type}"
        
        # Add validations randomly
        if random.random() > 0.6:
            validations = random.sample(VALIDATIONS, k=random.randint(1, 3))
            field_desc += f", {', '.join(validations)}"
        
        field_desc += ")"
        fields.append(field_desc)
    
    fields_str = ", ".join(fields)
    
    # Select a template and fill it
    template = random.choice(FORM_TEMPLATES)
    if "{sections}" in template:
        if sections_str:
            request = template.format(form_type=form_type, sections=sections_str, fields=fields_str)
        else:
            # Use a different template if we didn't generate sections
            template = "Create a {form_type} form with fields for {fields}."
            request = template.format(form_type=form_type, fields=fields_str)
    else:
        request = template.format(form_type=form_type, fields=fields_str, validations=", ".join(random.sample(VALIDATIONS, k=2)))
    
    return request

def generate_form_modification_request(form: Dict) -> str:
    """Generate a request to modify an existing form."""
    # Different types of modifications
    mod_types = ["add", "remove", "update", "change"]
    
    # Choose a modification type
    mod_type = random.choice(mod_types)
    
    if mod_type == "add":
        # Add a field to an existing section
        if form["formSections"]:
            section = random.choice(form["formSections"])
            field_type = random.choice(list(FIELDS.keys()))
            field_name = random.choice(FIELDS[field_type])
            return f"Add a {field_type} field for {field_name} to the {section['sectionTitle']} section"
        else:
            # No sections yet, add a section
            section_name = random.choice(SECTIONS)
            return f"Add a new section called {section_name}"
    
    elif mod_type == "remove":
        # Remove a field or section
        if form["formSections"]:
            section = random.choice(form["formSections"])
            if section.get("formControls", []):
                control = random.choice(section["formControls"])
                return f"Remove the {control['label']} field from the {section['sectionTitle']} section"
            else:
                return f"Delete the {section['sectionTitle']} section"
        else:
            # Nothing to remove, change to an add request
            return generate_form_modification_request({"formSections": []})
    
    elif mod_type == "update":
        # Update validation for a field
        if form["formSections"]:
            sections_with_controls = [s for s in form["formSections"] if s.get("formControls", [])]
            if sections_with_controls:
                section = random.choice(sections_with_controls)
                control = random.choice(section["formControls"])
                validation = random.choice(VALIDATIONS)
                return f"Make the {control['label']} field in the {section['sectionTitle']} section {validation}"
            else:
                # No controls, add one
                return generate_form_modification_request(form)
        else:
            # No sections, add one
            return generate_form_modification_request({"formSections": []})
    
    elif mod_type == "change":
        # Change the form title
        new_title = f"{random.choice(['New', 'Updated', 'Revised'])} {random.choice(FORM_TYPES).title()} Form"
        return f"Change the form title to '{new_title}'"
    
    return "Add a new section with contact information fields"

def execute_tools_for_prompt(prompt: str) -> List[Dict]:
    """Take a prompt and execute it through our extraction logic to get tool calls."""
    # Reset the form builder
    builder.reset_form()
    
    # Extract sections and controls from the prompt
    extracted_data = extract_sections_and_controls(prompt)
    
    # Process the extracted data to generate expected tool calls
    expected_tool_calls = []
    
    # First set the form title
    form_type = None
    for form_type_candidate in FORM_TYPES:
        if form_type_candidate in prompt.lower():
            form_type = form_type_candidate
            break
    
    if form_type:
        title = f"{form_type.title()} Form"
        expected_tool_calls.append({
            "name": "set_form_title",
            "parameters": {"title": title}
        })
    
    # Process the extracted sections and controls
    for section_data in extracted_data:
        section_title = section_data["section"]
        expected_tool_calls.append({
            "name": "add_section",
            "parameters": {"sectionTitle": section_title}
        })
        
        # Add controls to the section
        for control_data in section_data["controls"]:
            validation = {}
            required = False
            
            # Convert extracted validations to the format expected by add_control
            for key, value in control_data["validations"].items():
                if key == "required":
                    required = True
                else:
                    validation[key] = value
            
            expected_tool_calls.append({
                "name": "add_control",
                "parameters": {
                    "sectionTitle": section_title,
                    "controlType": control_data["type"],
                    "label": control_data["field"],
                    "required": required,
                    "validation": validation if validation else None
                }
            })
    
    return expected_tool_calls

def generate_dataset(num_examples: int) -> List[Dict]:
    """Generate a dataset with the specified number of examples."""
    dataset = []
    
    # Track used prompts to avoid duplicates
    used_prompts = set()
    
    # Generate examples
    for _ in range(num_examples):
        # Generate a new form request prompt
        prompt = generate_random_form_request()
        
        # Ensure we don't repeat prompts
        while prompt in used_prompts:
            prompt = generate_random_form_request()
        
        used_prompts.add(prompt)
        
        # Generate expected tool calls
        expected_tool_calls = execute_tools_for_prompt(prompt)
        
        # Add to dataset in Alpaca format
        dataset.append({
            "instruction": "Build a form according to the requirements. Return the necessary tool calls.",
            "input": prompt,
            "output": json.dumps(expected_tool_calls, indent=2)  # Convert list to JSON string
        })
    
    # Generate some modification examples (about 20% of the dataset)
    form_state = None
    for _ in range(max(1, num_examples // 5)):
        if not form_state:
            # Generate a form state to base modifications on
            builder.reset_form()
            base_prompt = generate_random_form_request()
            execute_tools_for_prompt(base_prompt)
            form_state = builder.get_current_form()
        
        # Generate a modification request
        mod_prompt = generate_form_modification_request(form_state)
        
        tool_calls = [
            {
                "name": "add_control",
                "parameters": {
                    "sectionTitle": "Contact Information",
                    "controlType": "email",
                    "label": "Work Email",
                    "required": True
                }
            }
        ]
        
        dataset.append({
            "instruction": "Modify the existing form according to the requirement. Return the necessary tool calls.",
            "input": f"I have an existing form. {mod_prompt}.",
            "output": json.dumps(tool_calls, indent=2)
        })
    
    return dataset

# Generate dataset with 100 examples
dataset = generate_dataset(100)

# Save to JSON file
with open("form_builder_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Dataset with {len(dataset)} examples generated and saved to form_builder_dataset.json")