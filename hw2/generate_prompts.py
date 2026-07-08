import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Define paths
PROJECT_DIR = Path(__file__).parent
DATA_PATH = PROJECT_DIR / "data" / "medical_cases.json"
TEMPLATES_DIR = PROJECT_DIR / "templates"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# Create output directory if it does not exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Jinja environment
env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Load data
with DATA_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

project_name = data["project_name"]
examples = data["few_shot_examples"]
cases = data["cases"]

# Load template
template = env.get_template("simplification_prompt.txt.j2")

# Generate outputs
for case in cases:
    rendered_prompt = template.render(
        project_name=project_name,
        examples=examples,
        case=case
    )

    output_file = OUTPUTS_DIR / f"prompt_{case['case_id']}.txt"
    output_file.write_text(rendered_prompt, encoding="utf-8")

print(f"Generated {len(cases)} prompt files in {OUTPUTS_DIR.resolve()}")