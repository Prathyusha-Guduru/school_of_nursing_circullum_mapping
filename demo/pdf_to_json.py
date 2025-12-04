# demo/pdf_to_json.py

import os
import json
from typing import List, Optional

from PyPDF2 import PdfReader
from pydantic import BaseModel, Field
from openai import OpenAI
import instructor

# ----------------------------
# CONFIG
# ----------------------------

MODEL = "gpt-4o-mini"
API_KEY = os.getenv("OPENAI_API_KEY")  # REQUIRED
client = instructor.from_openai(OpenAI(api_key=API_KEY))

PDF_FOLDER = "syllabi_pdfs"
AACN_FOLDER = "aacn_pdfs"
OUTPUT_SYLLABI_JSON = "syllabi_extracted.json"
OUTPUT_AACN_JSON = "aacn_domain_consolidated.json"


# ----------------------------
# HELPERS
# ----------------------------

# demo/pdf_to_json.py (ADD THESE)

def extract_syllabi_from_uploaded(files):
    results = []
    for file in files:
        text = extract_text_from_pdf(file.name)  # file.name is temp filepath
        obj = extract_syllabus_info(text, file.name.split("/")[-1])
        if obj:
            d = obj.model_dump()
            d["source_file"] = file.name
            results.append(d)
    return results


def extract_aacn_from_uploaded(files):
    results = []
    for file in files:
        text = extract_text_from_pdf(file.name)
        obj = consolidate_aacn_pis(text, file.name.split("/")[-1])
        if obj:
            d = obj.model_dump()
            d["source_file"] = file.name
            results.append(d)
    return results


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"PDF read error: {e}")
    return text.strip()


# ----------------------------
# SYLLABUS SCHEMA
# ----------------------------

class SyllabusData(BaseModel):
    course_title: Optional[str] = None
    course_description: Optional[str] = None
    learning_objectives: List[str] = Field(default_factory=list)
    learning_outcomes: List[str] = Field(default_factory=list)
    assessment_methods: List[str] = Field(default_factory=list)
    topical_outline: List[str] = Field(default_factory=list)


# ----------------------------
# SYLLABI EXTRACTION
# ----------------------------

def extract_syllabus_info(text: str, filename: str) -> Optional[SyllabusData]:
    if not text:
        return None

    prompt = f"""
    Extract structured syllabus data for curriculum analysis.

    ONLY return JSON following the schema.

    Syllabus text ({filename}):
    \"\"\"{text}\"\"\"
    """

    try:
        syllabus_obj = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Return strict JSON."},
                {"role": "user", "content": prompt}
            ],
            response_model=SyllabusData,
            temperature=0
        )
        return syllabus_obj

    except Exception as e:
        print(f"Syllabus LLM extraction failed: {e}")
        return None


def run_syllabus_extraction():
    syllabi_data = []
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    for pdf in pdf_files:
        path = os.path.join(PDF_FOLDER, pdf)
        text = extract_text_from_pdf(path)
        obj = extract_syllabus_info(text, pdf)

        if obj:
            d = obj.model_dump()
            d["source_file"] = pdf
            syllabi_data.append(d)
        else:
            syllabi_data.append({"source_file": pdf, "status": "FAILED"})

    with open(OUTPUT_SYLLABI_JSON, "w", encoding="utf-8") as f:
        json.dump(syllabi_data, f, indent=2)

    return syllabi_data


# ----------------------------
# AACN CONSOLIDATION
# ----------------------------

class Domain1ConsolidatedPIs(BaseModel):
    domain_name: str
    progression_indicators: List[str]


def consolidate_aacn_pis(text: str, filename: str) -> Optional[Domain1ConsolidatedPIs]:
    prompt = f"""
    Extract domain name and consolidate ALL progression indicators.

    Input ({filename}):
    \"\"\"{text}\"\"\"
    """

    try:
        obj = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Return strict JSON."},
                {"role": "user", "content": prompt}
            ],
            response_model=Domain1ConsolidatedPIs,
            temperature=0
        )
        return obj

    except Exception as e:
        print(f"AACN consolidation failed: {e}")
        return None


def run_aacn_consolidation():
    aacn_data = []
    pdf_files = [f for f in os.listdir(AACN_FOLDER) if f.lower().endswith(".pdf")]

    for pdf in pdf_files:
        path = os.path.join(AACN_FOLDER, pdf)
        text = extract_text_from_pdf(path)
        obj = consolidate_aacn_pis(text, pdf)

        if obj:
            d = obj.model_dump()
            d["source_file"] = pdf
            aacn_data.append(d)
        else:
            aacn_data.append({"source_file": pdf, "status": "FAILED"})

    with open(OUTPUT_AACN_JSON, "w", encoding="utf-8") as f:
        json.dump(aacn_data, f, indent=2)

    return aacn_data
