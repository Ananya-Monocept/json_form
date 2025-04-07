from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from trustcall import create_extractor
from ollama import Client as OllamaClient  # Assuming Ollama is used for LLM calls
import json
from form_builder.src.form_models import IForm, IFormSections, IFormControl, IValidator

# Load environment variables
load_dotenv()

app = FastAPI(title="Form Builder API with LangGraph")

# Initialize Ollama client
ollama_client = OllamaClient()
DEFAULT_MODEL = "llama3.2:3b-instruct-q8_0"

# Define tools for TrustCall
tools = [
    {
        "name": "add_section",
        "description": "Adds a new section to the form.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {"type": "string", "description": "Title of the section"}
            },
            "required": ["sectionTitle"]
        }
    },
    {
        "name": "add_control",
        "description": "Adds a new control to a section.",
        "parameters": {
            "type": "object",
            "properties": {
                "sectionTitle": {"type": "string", "description": "Title of the section"},
                "controlType": {"type": "string", "description": "Type of the control"},
                "label": {"type": "string", "description": "Label of the control"},
                "required": {"type": "boolean", "description": "Whether the control is required"},
                "validation": {"type": "object", "description": "Validation rules for the control"}
            },
            "required": ["sectionTitle", "controlType", "label"]
        }
    },
    {
        "name": "update_control",
        "description": "Updates an existing control.",
        "parameters": {
            "type": "object",
            "properties": {
                "controlName": {"type": "string", "description": "Name of the control"},
                "updates": {"type": "object", "description": "Key-value pairs to update"}
            },
            "required": ["controlName", "updates"]
        }
    },
    {
        "name": "delete_control",
        "description": "Deletes an existing control.",
        "parameters": {
            "type": "object",
            "properties": {
                "controlName": {"type": "string", "description": "Name of the control"}
            },
            "required": ["controlName"]
        }
    }
]

# Initialize TrustCall extractor
trustcall = create_extractor(
    llm=ollama_client,
    tools=tools,
    tool_choice="any"
)

# FormBuilder Class
class FormBuilder:
    def __init__(self):
        self.form = IForm(
            formTitle="Untitled Form",
            themeFile="default_theme.json",
            formSections=[]
        )

    def add_section(self, section_title: str) -> Dict:
        section_id = f"section_{len(self.form.formSections)}"
        new_section = IFormSections(
            sectionTitle=section_title,
            formControls=[]
        )
        self.form.formSections.append(new_section)
        return {"status": "success", "section_id": section_id}

    def add_control(self, section_title: str, control_type: str, label: str, required: bool = False, validation: Optional[Dict] = None) -> Dict:
        for section in self.form.formSections:
            if section.sectionTitle == section_title:
                control = IFormControl(
                    name=f"control_{len(section.formControls)}",
                    label=label,
                    visibleLabel=True,
                    type_=control_type,
                    validators=[IValidator(required=required, **(validation or {}))]
                )
                section.formControls.append(control)
                return {"status": "success", "control_name": control.name}
        return {"error": "Section not found"}

    def update_control(self, control_name: str, updates: Dict) -> Dict:
        for section in self.form.formSections:
            for control in section.formControls:
                if control.name == control_name:
                    for key, value in updates.items():
                        setattr(control, key, value)
                    return {"status": "updated"}
        return {"error": "Control not found"}

    def delete_control(self, control_name: str) -> Dict:
        for section in self.form.formSections:
            section.formControls = [c for c in section.formControls if c.name != control_name]
        return {"status": "deleted"}

    def get_form(self) -> Dict:
        return self.form.model_dump()

# Initialize FormBuilder
builder = FormBuilder()

# Define the state model for the graph
class FormState(BaseModel):
    prompt: str
    tool_calls: List[Dict] = []
    form_data: Dict[str, Any] = {}

# Define the graph
graph = StateGraph(FormState)

# Step 1: Parse the prompt and generate tool calls
def parse_prompt(state: FormState) -> FormState:
    result = trustcall.invoke({
        "messages": [{"role": "user", "content": state.prompt}]
    })
    tool_calls = result.get("tool_calls", [])
    return FormState(prompt=state.prompt, tool_calls=tool_calls)

graph.add_node("parse_prompt", parse_prompt)

# Step 2: Execute tool calls
def execute_tools(state: FormState) -> FormState:
    for call in state.tool_calls:
        func = getattr(builder, call["name"], None)
        if func:
            func(**call["parameters"])
    return state

graph.add_node("execute_tools", execute_tools)

# Step 3: Generate the final form schema
def generate_form(state: FormState) -> FormState:
    form_schema = builder.get_form()
    return FormState(prompt=state.prompt, form_data=form_schema)

graph.add_node("generate_form", generate_form)

# Connect the nodes
graph.add_edge(START, "parse_prompt")
graph.add_edge("parse_prompt", "execute_tools")
graph.add_edge("execute_tools", "generate_form")
graph.add_edge("generate_form", END)

# Compile the graph
app_graph = graph.compile()

@app.post("/generate-form")
async def generate_form(prompt: str) -> Dict:
    """
    Generate a form schema using LangGraph and TrustCall.
    """
    try:
        # Initialize the state
        initial_state = FormState(prompt=prompt)

        # Run the graph
        final_state = app_graph.invoke(initial_state)

        # Return the generated form
        return final_state.form_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-section")
async def add_section(section_title: str) -> Dict:
    return builder.add_section(section_title)

@app.post("/add-control")
async def add_control(
    section_title: str,
    control_type: str,
    label: str,
    required: bool = False,
    validation: Optional[Dict] = None
) -> Dict:
    return builder.add_control(section_title, control_type, label, required, validation)

@app.put("/update-control/{control_name}")
async def update_control(control_name: str, updates: Dict) -> Dict:
    return builder.update_control(control_name, updates)

@app.delete("/delete-control/{control_name}")
async def delete_control(control_name: str) -> Dict:
    return builder.delete_control(control_name)

@app.get("/get-form")
async def get_form() -> Dict:
    return builder.get_form()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)