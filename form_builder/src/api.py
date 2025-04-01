from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from form_builder.form_builder import FormBuilder, FormGenerator, validate_form
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Form Builder API")
builder = FormBuilder()
generator = FormGenerator()

class FormRequest(BaseModel):
    prompt: str
    method: str = "tools"  # "tools" or "direct"

class ToolCall(BaseModel):
    name: str
    parameters: Dict

@app.post("/generate-form")
async def generate_form(request: FormRequest) -> Dict:
    try:
        if request.method == "direct":
            # Use direct form generation
            form_json = generator.generate_with_jsonformer(request.prompt)
            # The validate_form function now needs 2 parameters
            return form_json
        else:
            # Use tool-based generation
            tool_calls = generator.generate(request.prompt, tools=["add_section", "add_control", "delete_control", "update_control"])
            results = []
            for call in tool_calls:
                func = getattr(builder, call["name"])
                result = func(**call["parameters"])
                results.append(result)
            return builder.get_form()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-section")
async def add_section(title: str) -> Dict:
    result = builder.add_section(title)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/add-control")
async def add_control(
    section_id: str,
    control_type: str,
    label: str,
    required: bool = False,
    validation: Optional[Dict] = None
) -> Dict:
    result = builder.add_control(
        section_id=section_id,
        control_type=control_type,
        label=label,
        required=required,
        validation=validation
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.put("/update-control/{control_id}")
async def update_control(control_id: str, updates: Dict) -> Dict:
    result = builder.update_control(control_id, **updates)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.delete("/delete-control/{control_id}")
async def delete_control(control_id: str) -> Dict:
    result = builder.delete_control(control_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/get-form")
async def get_form() -> Dict:
    return builder.get_form()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 