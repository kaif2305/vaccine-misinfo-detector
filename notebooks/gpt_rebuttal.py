{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5d78247d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m A new release of pip is available: \u001b[0m\u001b[31;49m25.0.1\u001b[0m\u001b[39;49m -> \u001b[0m\u001b[32;49m25.1.1\u001b[0m\n",
      "\u001b[1m[\u001b[0m\u001b[34;49mnotice\u001b[0m\u001b[1;39;49m]\u001b[0m\u001b[39;49m To update, run: \u001b[0m\u001b[32;49mpip install --upgrade pip\u001b[0m\n"
     ]
    }
   ],
   "source": [
    "!pip install openai python-dotenv --quiet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "477111e7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from dotenv import load_dotenv\n",
    "import os\n",
    "load_dotenv()\n",
    "\n",
    "from openai import OpenAI\n",
    "client = OpenAI(api_key=os.getenv(\"OPENAI_API_KEY\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ba0271eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "load_dotenv()  \n",
    "client = OpenAI(api_key=os.getenv(\"OPENAI_API_KEY\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "57e6c3dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "def build_prompt(claim: str) -> str:\n",
    "    prompt_path = Path(\"../src/rebuttal/prompts.txt\")  # ← 从 notebooks/ 向上一级，再进 src/rebuttal/\n",
    "    \n",
    "    if not prompt_path.exists():\n",
    "        raise FileNotFoundError(f\"prompts.txt not found at {prompt_path}\")\n",
    "\n",
    "    with open(prompt_path, 'r') as file:\n",
    "        template = file.read()\n",
    "\n",
    "    return template.replace('{claim}', claim)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "9bf9c513",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_rebuttal(claim: str) -> str:\n",
    "    \"\"\"\n",
    "    Given a claim, call OpenAI ChatCompletion API and return the response.\n",
    "    Includes basic fallback if response is incomplete.\n",
    "    \"\"\"\n",
    "    prompt = build_prompt(claim)\n",
    "\n",
    "    try:\n",
    "        response = client.chat.completions.create(\n",
    "            model=\"gpt-3.5-turbo\",  # or \"gpt-4\"\n",
    "            messages=[\n",
    "                {\"role\": \"system\", \"content\": \"You are a helpful medical fact-checking assistant.\"},\n",
    "                {\"role\": \"user\", \"content\": prompt}\n",
    "            ],\n",
    "            temperature=0.5,\n",
    "            max_tokens=300\n",
    "        )\n",
    "\n",
    "        reply = response.choices[0].message.content.strip()\n",
    "\n",
    "        if \"Verdict\" not in reply or \"Explanation\" not in reply:\n",
    "            return \"⚠️ Could not generate a reliable rebuttal.\"\n",
    "        \n",
    "        return reply\n",
    "\n",
    "    except Exception as e:\n",
    "        return f\"⚠️ API error: {str(e)}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e1890943",
   "metadata": {},
   "outputs": [],
   "source": [
    "claims = [\n",
    "    \"Vaccines contain microchips for tracking people.\",\n",
    "    \"COVID vaccines cause infertility in women.\",\n",
    "    \"mRNA vaccines change your DNA.\",\n",
    "    \"You don't need vaccines if you take enough vitamins.\",\n",
    "    \"Vaccinated people shed spike proteins and harm others.\",\n",
    "    \"COVID-19 was planned to sell vaccines.\",\n",
    "    \"Natural immunity is always better than vaccine immunity.\",\n",
    "    \"COVID vaccines were developed too quickly to be safe.\",\n",
    "    \"5G networks caused the coronavirus pandemic.\",\n",
    "    \"The vaccine has a magnetic effect on your body.\"\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a9876eba",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_and_save_rebuttals(claims, output_path='rebuttal_outputs.txt'):\n",
    "    \"\"\"\n",
    "    Given a list of claims, generate rebuttals and save them to a text file.\n",
    "    \"\"\"\n",
    "    all_outputs = []\n",
    "    \n",
    "    for i, claim in enumerate(claims, 1):\n",
    "        print(f\"📝 Generating for claim {i}/{len(claims)}...\")\n",
    "        response = generate_rebuttal(claim)\n",
    "        formatted = f\"Claim {i}: {claim}\\nRebuttal:\\n{response}\\n{'-'*60}\\n\"\n",
    "        all_outputs.append(formatted)\n",
    "    \n",
    "    with open(output_path, 'w') as f:\n",
    "        f.writelines(all_outputs)\n",
    "    \n",
    "    print(f\"\\n✅ Finished. All rebuttals saved to {output_path}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "3b16ebdb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📝 Generating for claim 1/10...\n",
      "📝 Generating for claim 2/10...\n",
      "📝 Generating for claim 3/10...\n",
      "📝 Generating for claim 4/10...\n",
      "📝 Generating for claim 5/10...\n",
      "📝 Generating for claim 6/10...\n",
      "📝 Generating for claim 7/10...\n",
      "📝 Generating for claim 8/10...\n",
      "📝 Generating for claim 9/10...\n",
      "📝 Generating for claim 10/10...\n",
      "\n",
      "✅ Finished. All rebuttals saved to rebuttal_outputs.txt\n"
     ]
    }
   ],
   "source": [
    "generate_and_save_rebuttals(claims)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fdbd70eb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
