"""
Rule-based rebuttal system for misinformation claims.
This system uses a simple keyword matching approach to identify claims
and provide rebuttals based on a predefined knowledge base (KB).

  Database example:

  [
    {
        "keywords": ["autism", "vaccine"],
        "rebuttal": "Multiple studies show no link between vaccines and autism.",
        "sources": [
            "https://www.cdc.gov/vaccinesafety/concerns/autism.html"
        ]
    },
    {
        "keywords": ["mRNA", "change DNA"],
        "rebuttal": "mRNA vaccines do not alter human DNA.",
        "sources": [
            "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/facts.html"
        ]
    }
]
  
@Author: Xinrui WAN
@Date: 03/May/2025
"""
import json
from typing import List, Dict, Optional

class RebuttalRetriever:
    """
    A simple rule-based rebuttal system for misinformation claims.
    """
    def __init__(self, kb_path: str):
        self.kb_path = kb_path
        self.knowledge_base = self._load_kb()

    def _load_kb(self) -> List[Dict]:
        """
        Load the knowledge base from a JSON file.
        :return: List of dictionaries containing keywords, rebuttals, and sources.
        """
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise Exception(f"Error loading knowledge base: {e}")

    def _match_keywords(self, claim: str, keywords: List[str]) -> bool:
        """
        Check if all keywords are present in the claim.
        :param claim: The claim to be checked.
        :param keywords: List of keywords to match.
        :return: True if all keywords are found in the claim, otherwise False.
        """
        claim_lower = claim.lower()
        return all(keyword.lower() in claim_lower for keyword in keywords)

    def get_rebuttal(self, claim: str) -> Optional[Dict]:
        """
        Given a claim, return the rebuttal and sources from the knowledge base.
        :param claim: The claim to be checked.
        :return: A dictionary containing the rebuttal and sources if found, otherwise None.
        """
        for entry in self.knowledge_base:
            if self._match_keywords(claim, entry.get("keywords", [])):
                return {
                    "rebuttal": entry.get("rebuttal", ""),
                    "sources": entry.get("sources", [])
                }
        return None


# Example usage
if __name__ == "__main__":
    retriever = RebuttalRetriever("../../data/rebuttal/rebuttal_kb.json")

    test_claims = [
        "Vaccines cause autism in children.",
        "They inject microchips through COVID vaccines.",
        "Does mRNA change your DNA?",
        "Vaccines might reduce fertility."
    ]

    for claim in test_claims:
        result = retriever.get_rebuttal(claim)
        print(f"\nClaim: {claim}")
        if result:
            print(f"Rebuttal: {result['rebuttal']}")
            print(f"Sources: {', '.join(result['sources'])}")
        else:
            print("No rebuttal found.")
