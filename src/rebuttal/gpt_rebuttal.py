from dotenv import load_dotenv
import os
from openai import OpenAI
from pathlib import Path


class VaccineRebuttalGenerator:
    def __init__(self, prompt_path: str = r"../../data/rebuttal/prompts.txt"):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #self.client = OpenAI(api_key="########") #TEST Replace with your actual API key, FOR TESTING ONLY
        self.prompt_path = Path(prompt_path)

    def build_prompt(self, claim: str) -> str:
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"prompts.txt not found at {self.prompt_path}")

        with open(self.prompt_path, 'r') as file:
            template = file.read()

        return template.replace('{claim}', claim)

    def generate_rebuttal(self, claim: str) -> str:
        """
        Given a claim, call OpenAI ChatCompletion API and return the response.
        Includes basic fallback if response is incomplete.
        """
        prompt = self.build_prompt(claim)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful medical fact-checking assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )

            reply = response.choices[0].message.content.strip()

            if "Verdict" not in reply or "Explanation" not in reply:
                raise Exception("Could not generate a reliable rebuttal.")

            return reply

        except Exception as e:
            raise Exception(f"API error: {str(e)}")

    def generate_rebuttals(self, claims : list[str]) -> str:
        """
        Given a list of claims, yield rebuttals.
        """
        for claim in claims:
            response = self.generate_rebuttal(claim)
            yield response



if __name__ == "__main__":
    claims = [
        "Vaccines contain microchips for tracking people.",
        "COVID vaccines cause infertility in women.",
        "mRNA vaccines change your DNA.",
        "You don't need vaccines if you take enough vitamins.",
        "Vaccinated people shed spike proteins and harm others.",
        "COVID-19 was planned to sell vaccines.",
        "Natural immunity is always better than vaccine immunity.",
        "COVID vaccines were developed too quickly to be safe.",
        "5G networks caused the coronavirus pandemic.",
        "The vaccine has a magnetic effect on your body."
    ]

    generator = VaccineRebuttalGenerator()
    for rebuttal in generator.generate_rebuttals(claims):
        print(rebuttal)

