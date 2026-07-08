# ANLP_2026

## Homework 2 Part 5

### Medical Text Simplification and Translation Demo

### Task and Domain
The domain is medical communication. The task involves utilizing a Large Language Model (LLM) to simplify complex medical diagnoses and treatment plans for specific patient demographics. 

### Function of the Jinja Template
The Jinja template (`simplification_prompt.txt.j2`) dynamically generates LLM instructions. It ingests a structured JSON configuration and outputs a formatted text prompt. It adjusts the task instructions based on:
- Target audience age group (e.g., child, teenager, adult layperson).
- Target language for translation.
- Boolean flags indicating whether to extract a structured glossary of medical terms.
- Inclusion of few-shot examples for specific formatting.

### Justification for Jinja
A templating engine is required for this task because the prompt configuration relies heavily on conditional formatting and schema alterations. Relying on native Python string formatting or multiple `if/else` concatenation blocks would yield unmaintainable code. Jinja provides distinct advantages here:
- **Conditional Logic:** Instructions for tone (e.g., speaking to a child versus an adult) and translation are injected only when specific metadata criteria are met.
- **Loops:** Few-shot examples are iterated over cleanly to construct the context window.
- **Output Formats:** The JSON schema definition required for the LLM output changes dynamically based on the `extract_glossary` parameter.