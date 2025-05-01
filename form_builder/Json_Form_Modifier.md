
# Project Overview

This project is a **form builder API** built using FastAPI, LangGraph, and Google Gemini. It allows users to:

- Generate forms via text prompts
- Modify existing forms (from JSON or templates)
- Dynamically add/remove/update form sections and controls
- Save/load forms as templates
- Handle validation, dependent controls, and dynamic control groups

It uses an LLM (Gemini) to interpret natural language prompts into structured tool calls that modify the form state.

# Core Components

### 1. **FormBuilder Class**

Acts as the central engine for building and managing the form structure. It:

- Manages sections and controls
- Handles validation, visibility rules, dependencies
- Provides methods like `add_section`, `add_control`, `delete_control`, etc.

### 2. **Tools Dictionary**

A dictionary mapping tool names to functions and metadata. Used by the LLM agent to perform actions on the form.

### 3. **LangGraph Workflow**

Defined using `langgraph.Graph()` with multiple nodes that handle different aspects of form generation/modification.


# FormBuilder Class Functions

### ✅ **Initialization & Reset**

- `__init__(self)`: Initializes the form builder, sets up form state and tracking counters.
- `reset_form(self)`: Resets the form to default (empty) state.
- `set_expected_controls(self, count: int)`: Sets the expected number of controls for completeness check.

---

### 🧩 **Form Structure Management**

#### 🔲 Sections

- `add_section(self, sectionTitle: str) -> Dict`: Adds a new section or uses an existing one if already exists.

### ⚙️ Controls

#### add_control 
    Adds a control (like text, radio, checkbox, etc.) to a specified section.  
    Parameters:
    - `sectionTitle`, `controlType`, `label`, `name`, `required`, `validation`, `dependentControls`, `radioOptions`, `visibilityRules`, `visible`.

#### 🗑️ Deletion

- `delete_section(self, sectionTitle: str) -> Dict`: Deletes a specified section from the form.
- `delete_control(self, sectionTitle: str, controlName: str, is_dependent: bool = False) -> Dict`: Deletes a control and all references in the form.

---

#### 🔁 Dynamic Controls

- `add_dynamic_controls(...)` → Adds dynamic control groups under a parent control.
- `update_dynamic_control(...)` → Updates a specific dynamic control.
- `delete_dynamic_control(...)` → Deletes a specific dynamic control within a group.

---

#### 🔐 Validation

- `update_control_validation(self, sectionTitle: str, controlName: str, validation: Dict) -> Dict`: Updates or adds validation rules (e.g., required, minLength, maxLength, pattern).

---

#### 🏷️ Form Meta Info

- `set_form_title(self, title: str) -> Dict`: Sets the form’s title.

---

#### 🔄 State & Completeness Checks

- `is_form_complete(self) -> bool`: Checks whether the form has reached the expected number of controls or structure.
- `remove_duplicate_controls(self) -> Dict`: Removes duplicate controls that may have been added unintentionally.
- `get_current_form(self) -> Dict`: Returns the current form state in a cleaned format (without nulls, properly formatted).
- `transform_type_back(self, data: Any)`: Recursively transforms `'type_'` back to `'type'` for JSON compatibility.

---

#### 🔄 Helper Methods for Internal Logic

- `_add_dependent_controls(...)`: Adds dependent controls to a section based on visibility rules.
- `_update_dependent_controls(...)`: Updates or deletes dependent controls associated with radio options.
- `get_dependent_controls(...)`: Retrieves dependent control information for a given control.
- `_clean_control_references(...)`: Cleans up any remaining references to deleted controls across the form.

---

#### 🧹 Data Transformation Helpers

- `load_form_data(self, form_data: Dict) -> Dict`: Loads and initializes form data into the form builder from a JSON input.



# Available Tools

![[Pasted image 20250501122105.png]]
# LangGraph Workflow Nodes

### 1. `start_node`

#### 📍 Purpose:

Initialize the workflow based on user input.

#### 🧠 Functions Used:

- `detect_request_type()` (async): Determines if the request is to:
    - Create a **new form** , or
    - Modify an **existing form** .
- `extract_sections_and_controls()`: Parses sections and controls from the prompt for new forms.
- Sets up a system message for the LLM based on whether it's a new or modification request.

#### 📌 Key Logic:

- If it's a **modification** , it uses the current form state to construct a prompt showing existing sections/controls.
- If it's a **new form** , it dynamically extracts sections and fields using regex patterns and sets up a prompt asking LLM to generate tool calls.
- Returns initial messages and flags like `is_modification` for downstream logic.

---

### 2. `llm_node`

#### 📍 Purpose:

Use Google Gemini LLM to interpret the instruction and return structured tool calls.

#### 🧠 Functions Used:

- Uses `Gemini` model to send messages and receive responses.
- `extract_tool_calls_from_message()`: Tries multiple methods to extract JSON-formatted tool calls:
    - Full JSON parsing
    - Regex-based extraction
- Validates and filters duplicate operations.

#### 📌 Key Logic:

- Limits iterations to prevent infinite loops (max: 20).
- Sends prompt and history to LLM.
- Parses response into valid tool calls.
- Tracks successful/failed operations.
- Determines if the form is complete (`form_complete`) based on expected control count or iteration limit.

---

### 3. `tool_node`

#### 📍 Purpose:

Execute the actual actions (like adding/deleting controls) on the form builder based on LLM-generated tool calls.

#### 🧠 Functions Used:

- Iterates through `tool_calls` and executes them via `tools[tool_name]["func"](**parameters)`
- Handles success/failure tracking and error feedback.
- After execution, appends results back as a tool message.
- Calls `remove_duplicate_controls()` to clean up any possible duplicates created during modifications.

#### 📌 Key Logic:

- Logs each tool execution with parameters.
- On failure, returns helpful context to guide the next LLM call.
- Builds the updated form state after all tools executed.
- Decides if the form is complete based on:
    - Successful execution of all tool calls
    - Hitting iteration limits

---

### 4. `end_node`

#### 📍 Purpose:

Finalize the form generation by deduplicating controls and returning the final form.

#### 🧠 Functions Used:

- `builder.remove_duplicate_controls()`: Removes duplicate controls within sections.
- Retrieves the final form state using `builder.get_current_form()`.

#### 📌 Key Logic:

- Cleans up post-processing artifacts.
- Returns the final form structure along with metadata like duplicates removed.
- Marks the form as completed.


![[Pasted image 20250501121005.png]]


# How to add new tools

##### FLOW:
FormBuilder logic → API endpoint (optional) → tools registry → LLM instruction → Workflow logic (if needed)

##### STEP BY STEP PROCESS

- Add a new method inside the `FormBuilder` class that performs the desired action.
- Register the function under the `tools` dictionary with metadata like description and parameters. 
   This is the tools dictionary:

     ![[Pasted image 20250501122932.png]]
- Update system prompts or instructions in `start_node` or elsewhere so the LLM knows about the tool's availability.

