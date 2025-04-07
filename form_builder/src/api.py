from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from ollama import Client as OllamaClient
import json
from langgraph.graph import START, END, Graph
from form_models import IForm, IFormSections, IFormControl, IValidator

# Load environment variables
load_dotenv()

app = FastAPI(title="Form Builder API")

# Define a fallback model
DEFAULT_MODEL = "llama3.2:3b-instruct-q8_0"

# Initialize Ollama client
ollama_client = OllamaClient()


class FormBuilder:
    def __init__(self):
        self.reset_form()

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

    def add_section(self, sectionTitle: str) -> Dict:
        """Add a new section to the form."""
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

    def add_control(self, sectionTitle: str, controlType: str, label: str, required: bool = False, validation: Optional[Dict] = None) -> Dict:
        """Add a new control to a section."""
        for section in self.form.formSections:
            if section.sectionTitle == sectionTitle:
                control = IFormControl(
                    name=f"control_{len(section.formControls)}",
                    label=label,
                    visibleLabel=True,
                    type_=controlType,
                    validators=[IValidator(required=required, **(validation or {}))] if required or validation else None
                )
                section.formControls.append(control)
                return {"status": "success", "control_name": control.name}
        return {"error": "Section not found"}

    def set_form_title(self, title: str) -> Dict:
        """Set the form title."""
        self.form.formTitle = title
        return {"status": "success", "form_title": title}

    def is_form_complete(self) -> bool:
        """Check if the form has at least one section with controls."""
        return len(self.form.formSections) > 0 and any(
            len(section.formControls) > 0 for section in self.form.formSections
        )

    def get_current_form(self) -> Dict:
        """Return the current form state"""
        try:
            return self.form.model_dump()
        except AttributeError:
            return self.form.dict()


# Initialize FormBuilder
builder = FormBuilder()


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


# Define the workflow
workflow = Graph()


@workflow.add_node
def start_node(state: dict):
    """Process the initial user input."""
    system_prompt = """You are a form generation assistant. Your task is to create forms based on user requirements.
    You have access to these tools:
    1. add_section - Use this to add a new section to the form
    2. add_control - Use this to add form fields to sections
    3. set_form_title - Use this to set a descriptive title for the form
    
    Follow these steps for every form request:
    1. First set an appropriate form title using set_form_title
    2. Create appropriate sections using add_section
    3. Add relevant controls to each section using add_control
    
    For control types, choose from: text, email, number, date, select, textarea, checkbox, radio
    
    Example response format:
    [
      {
        "name": "set_form_title", 
        "parameters": {"title": "Personal Information Form"}
      },
      {
        "name": "add_section",
        "parameters": {"sectionTitle": "Basic Information"}
      },
      {
        "name": "add_control",
        "parameters": {
          "sectionTitle": "Basic Information",
          "controlType": "text",
          "label": "Full Name",
          "required": true
        }
      }
    ]
    
    Make sure to create a complete and usable form with all necessary fields based on the user's request.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["prompt"]}
        ],
        "iteration_count": 0
    }


@workflow.add_node
def llm_node(state: dict):
    """Call the LLM to determine actions."""
    messages = state["messages"]
    iteration_count = state.get("iteration_count", 0)
    
    # Prevent infinite loops
    if iteration_count > 10:
        return {
            "messages": messages,
            "iteration_count": iteration_count,
            "tool_calls": [],
            "form_complete": True
        }

    # Prepare the prompt for the LLM
    prompt = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "format": "json",
        "options": {"temperature": 0.3}
    }

    # Get response from Ollama
    try:
        response = ollama_client.chat(**prompt)
        message = response["message"]
        
        # Try to parse tool calls from the response
        tool_calls = []
        try:
            content = json.loads(message["content"])
            if isinstance(content, list):
                tool_calls = content
            elif isinstance(content, dict):
                if "tool_calls" in content:
                    tool_calls = content["tool_calls"]
                elif all(k in content for k in ["name", "parameters"]):
                    tool_calls = [content]
        except (json.JSONDecodeError, AttributeError, TypeError):
            # If parsing fails, assume it's a natural language response
            print(f"LLM response not in expected format: {message['content']}")
            # Add the message to context and try again
            messages.append(message)
            return {
                "messages": messages,
                "tool_calls": [],
                "iteration_count": iteration_count + 1,
                "form_complete": False
            }

        return {
            "messages": messages + [message],
            "tool_calls": tool_calls,
            "iteration_count": iteration_count + 1,
            "form_complete": builder.is_form_complete()
        }
    except Exception as e:
        print(f"Error calling LLM: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@workflow.add_node
def tool_node(state: dict):
    """Execute the tools."""
    tool_calls = state["tool_calls"]
    messages = state["messages"]
    results = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tools:
            try:
                result = tools[tool_name]["func"](**tool_call["parameters"])
                print(f"Executed tool '{tool_name}' with result: {result}")
                results.append({
                    "tool_call_id": tool_call.get("id", ""),
                    "name": tool_name,
                    "result": result
                })
            except Exception as e:
                print(f"Error executing tool '{tool_name}': {e}")
                results.append({
                    "tool_call_id": tool_call.get("id", ""),
                    "name": tool_name,
                    "error": str(e)
                })

    return {
        "messages": messages + [{
            "role": "tool",
            "content": json.dumps(results),
            "tool_call_id": tool_call.get("id", "") if tool_calls else ""
        }],
        "results": results,
        "form_complete": builder.is_form_complete()
    }


@workflow.add_node
def end_node(state: dict):
    """Finalize the form generation."""
    return {"form": builder.form}


# Define the workflow edges
workflow.set_entry_point("start_node")
workflow.add_edge("start_node", "llm_node")

# After LLM node, decide whether to use tools or end
workflow.add_conditional_edges(
    "llm_node",
    lambda state: (
        state.get("form_complete", False) or 
        not state.get("tool_calls") or
        len(state.get("tool_calls", [])) == 0
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


# API Endpoints

@app.post("/generate-form")
async def generate_form(prompt: str) -> Dict:
    """Generate a form based on the user's prompt."""
    try:
        # Reset the form builder for a fresh form
        builder.reset_form()

        # Execute the workflow
        result = app_workflow.invoke({"prompt": prompt})

        # Return the generated form
        return builder.get_current_form()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-section")
async def add_section(title: str) -> Dict:
    """Manually add a section to the form."""
    try:
        result = builder.add_section(sectionTitle=title)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-control")
async def add_control(
    section_title: str,
    control_type: str,
    label: str,
    required: bool = False
) -> Dict:
    """Manually add a control to a section."""
    try:
        result = builder.add_control(
            sectionTitle=section_title,
            controlType=control_type,
            label=label,
            required=required
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set-title")
async def set_title(title: str) -> Dict:
    """Manually set the form title."""
    try:
        result = builder.set_form_title(title=title)
        return {"status": "success", "result": result}
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)