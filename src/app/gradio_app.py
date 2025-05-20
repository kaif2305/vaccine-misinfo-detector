import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import gradio as gr
from pathlib import Path
from inference.predict import VaccineMisinformationDetector
from rebuttal.gpt_rebuttal import VaccineRebuttalGenerator
from rebuttal.rule_rebuttal import RebuttalRetriever
from ocr.extract_text import extract_text_from_image 
from PIL import Image
from typing import Union



PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "bert_saved"
PROMPT_PATH = PROJECT_ROOT / "data" / "rebuttal" / "prompts.txt"
KB_PATH = PROJECT_ROOT / "data" / "rebuttal" / "rebuttal_kb.json"

classifier = VaccineMisinformationDetector(str(MODEL_PATH.resolve()))  # Convert to string
gpt_generator = VaccineRebuttalGenerator(prompt_path=PROMPT_PATH)
rule_retriever = RebuttalRetriever(KB_PATH)

def process_input(image_input: Union[Image.Image, None], text_input: Union[str, None]):
    """
    Handles both text and image inputs.
    """
    if image_input is not None:
        try:
            claim = extract_text_from_image(image_input)
        except Exception as e:
            return "OCR Failed", f"Error extracting text from image: {e}"
    elif text_input is not None:
        claim = text_input.strip()
    else:
        return "Invalid input", "Please provide either an image or text input."

    if not claim:
        return "No valid claim detected", "Please provide a valid text or image containing readable text."

    label = classifier.predict(claim)

    if label == 0:
        return "✅ This claim does not appear to be misinformation.", ""

    # Rule-based rebuttal
    rule_result = rule_retriever.get_rebuttal(claim)
    rule_text = "🔍 No rule-based rebuttal found."
    if rule_result:
        sources = "\n".join(f"- {src}" for src in rule_result["sources"])
        rule_text = f"📌 Rule-based Rebuttal:\n{rule_result['rebuttal']}\n\n🔗 Sources:\n{sources}"

    # GPT rebuttal
    try:
        gpt_text = gpt_generator.generate_rebuttal(claim)
    except Exception as e:
        gpt_text = f"⚠️ GPT Rebuttal generation failed: {e}"

    return "⚠️ This appears to be misinformation.", f"{rule_text}\n\n🧠 GPT Rebuttal:\n{gpt_text}"

# Gradio UI
iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Image(type="pil", label="Upload meme/claim image"), 
        gr.Textbox(label="Or enter a vaccine-related claim")
    ],
    outputs=[
        gr.Textbox(label="Classification"),
        gr.Textbox(label="Rebuttals")
    ],
    title="Vaccine Misinformation Detector (Text + Image)",
    description="Upload an image (e.g., meme) or enter a claim directly. The system detects vaccine misinformation and generates rebuttals."
)
if __name__ == "__main__":
    iface.launch()