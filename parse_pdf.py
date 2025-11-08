import os
import json
from PyPDF2 import PdfReader
from openai import OpenAI

# Initialize the client
client = OpenAI(api_key="")

# --- CONFIG ---
PDF_FOLDER = "syllabi"   # Folder containing your 30 PDFs
OUTPUT_FILE = "syllabi.json"
MODEL = "gpt-4o-mini"  # or "gpt-4o" if you have access

# --- Helper to extract text from PDF ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

# --- Helper to get structured JSON from GPT ---
def extract_syllabus_info(text, filename):
    prompt = f"""
You are an expert academic document parser.

Extract all **relevant structured data** from the following course syllabus.
Return ONLY valid JSON, no extra commentary.

Include (if present): 
- course_code
- course_name
- textbooks
- learning_objectives
- learning_outcomes
- grading_information
- instructor_information
- schedule
- assessment_methods
- policies
- any_other_relevant_fields

Syllabus text (from {filename}):
\"\"\"{text[:15000]}\"\"\"  # limit for token safety
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise data extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}  # ensures valid JSON output
    )
    return response.choices[0].message.content

# --- Main script ---
def main():
    syllabi_data = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"📄 Processing {pdf_file}...")

        text = extract_text_from_pdf(pdf_path)
        json_str = extract_syllabus_info(text, pdf_file)

        try:
            syllabus_info = json.loads(json_str)
            syllabus_info["source_file"] = pdf_file
            syllabi_data.append(syllabus_info)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: Could not parse JSON for {pdf_file}")
            syllabus_info = {"source_file": pdf_file, "raw_response": json_str}
            syllabi_data.append(syllabus_info)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(syllabi_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved structured syllabi data to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
