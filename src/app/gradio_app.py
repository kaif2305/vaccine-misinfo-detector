import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import gradio as gr
from pathlib import Path
from inference.predict import VaccineMisinformationDetector
from rebuttal.gpt_rebuttal import VaccineRebuttalGenerator
from rebuttal.rule_rebuttal import RebuttalRetriever



# Paths may need adjusting depending on your directory structure
PROMPT_PATH = Path("../data/rebuttal/prompts.txt")
KB_PATH = Path("../data/rebuttal/rebuttal_kb.json")

# Initialize once
MODEL_PATH = Path("../models/bert_saved")
classifier = VaccineMisinformationDetector(str(MODEL_PATH.resolve()))  # Convert to string
gpt_generator = VaccineRebuttalGenerator(prompt_path=PROMPT_PATH)
rule_retriever = RebuttalRetriever(KB_PATH)

def process_claim(claim):
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
    fn=process_claim,
    inputs=gr.Textbox(label="Enter a vaccine-related claim"),
    outputs=[
        gr.Textbox(label="Classification"),
        gr.Textbox(label="Rebuttals")
    ],
    title="Vaccine Misinformation Detector",
    description="Classifies claims and provides fact-based rebuttals using rule-based and GPT models."
)

if __name__ == "__main__":
    iface.launch()
