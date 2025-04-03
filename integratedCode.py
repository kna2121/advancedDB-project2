import requests
import sys
import time
from bs4 import BeautifulSoup
import spacy
from googleapiclient.discovery import build
import google.generativeai as genai
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
import torch
from spacy_help import get_entities, create_entity_pairs

# ======================= HARDCODED KEYS ============================
API_KEY = 'AIzaSyAYiEosxKFAa3cwpyN-Au3H7wRhZtAx8KY'
CX = 'd29acf7ff2d2f40a9'
GEMINI_API_KEY = 'AIzaSyBzTTV9qedmDGr-IjLhNYT9pzOAHpHOOIc'
genai.configure(api_key=GEMINI_API_KEY)

# ======================= RELATION CONFIGS ============================
relations = ['0', 'Schools_Attended', 'Work_For', 'Live_In', 'Top_Member_Employees']
relation_pairs = {
    1: ["PERSON", "ORGANIZATION"],
    2: ["PERSON", "ORGANIZATION"],
    3: ["PERSON", "LOCATION", "CITY", "COUNTRY", "STATE_OR_PROVINCE"],
    4: ["ORGANIZATION", "PERSON"]
}
relation_prompts = {
    1: "Identify any found school attended relationships in the sentence below. "
       "Return only if the subject is a person and the object is a school or university name. "
       "Return output in the format Person ; School on its own line.",
    2: "Identify any found work-for relationships in the sentence below. "
       "Return only if the subject is a person and the object is a place of work. "
       "Return output in the format Person ; Place of work on its own line.",
    3: "Identify any live-in relationships in the sentence below. "
       "Return only if the subject is a person and the object is a location (city or country). "
       "Return output in the format Person ; Location on its own line.",
    4: "Identify any top-member relationships in the sentence below. "
       "Return only if the subject is an organization and the object is a person. "
       "Return output in the format Organization ; Person on its own line."
}

# ======================= HELPERS ============================
def google_search(api_key, cx, query):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cx, num=10).execute()
    return [item['link'] for item in res.get("items", [])]

def fetch_page_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["style", "script", "head", "title", "table", "nav", "header"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)
        if len(text) > 10000:
            print(f"\tTrimming webpage content from {len(text)} to 10000 characters")
            text = text[:10000]
        print(f"\tWebpage length (num characters): {len(text)}")
        print("\tAnnotating the webpage using spaCy...")
        return text
    except Exception as e:
        print(f"\tUnable to fetch URL. Continuing...")
        return ""

def get_gemini(prompt, model_name, max_tokens, temperature, top_p, top_k):
    model = genai.GenerativeModel(model_name)
    config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    time.sleep(5)
    response = model.generate_content(prompt, generation_config=config)
    return response.text.strip() if response.text else "No response received"

# ======================= MAIN ============================
def main():
    method = sys.argv[1]
    r = int(sys.argv[2])
    t = float(sys.argv[3])
    seed_query = sys.argv[4]
    k = int(sys.argv[5])

    if method not in ['spanbert', 'gemini']:
        sys.exit("Method must be 'spanbert' or 'gemini'")
    if r not in relation_pairs:
        sys.exit("Relation must be an integer from 1 to 4.")
    if method == 'spanbert' and not (0 <= t <= 1):
        sys.exit("Threshold must be between 0 and 1.")
    if k <= 0:
        sys.exit("k must be > 0.")

    print(f"""
Parameters:
Client key     = {API_KEY}
Engine key     = {CX}
Gemini key     = {GEMINI_API_KEY}
Method         = {method}
Relation       = {relations[r]}
Threshold      = {t}
Query          = {seed_query}
# of Tuples    = {k}
""")

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    if method == 'spanbert':
        print("Loading SpanBERT model...")
        tokenizer = BertTokenizer.from_pretrained("./pretrained_spanbert")
        config = BertConfig.from_pretrained("./pretrained_spanbert")
        config.num_labels = 42
        model = BertForSequenceClassification(config)
        state_dict = torch.load("./pretrained_spanbert/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        with open("./pretrained_spanbert/relations.txt", "r") as f:
            labels = [line.strip() for line in f.readlines()]
        label_map = {i: label for i, label in enumerate(labels)}

    target_types = relation_pairs[r]
    prompt_template = relation_prompts[r]
    X = {}  # (subj, obj) → confidence
    processed, queried = set(), set()
    current_query = seed_query
    iteration = 0

    while len(X) < k:
        print(f"\n=========== Iteration: {iteration} - Query: {current_query} ===========")
        urls = google_search(API_KEY, CX, current_query)
        if not urls:
            print("No search results found.")
            break

        for idx, url in enumerate(urls):
            if url in processed:
                continue
            print(f"\nURL ({idx + 1}/{len(urls)}): {url}")
            print("\tFetching text from url...")
            raw_text = fetch_page_text(url)
            if not raw_text:
                continue

            doc = nlp(raw_text)
            sentences = list(doc.sents)
            print(f"\tExtracted {len(sentences)} sentences. Processing...")

            for sent in sentences:
                try:
                    pairs = create_entity_pairs(sent, target_types)
                    if not pairs:
                        continue

                    if method == 'gemini':
                        filtered = []
                        for _, e1, e2 in pairs:
                            e1_type, e2_type = e1[1], e2[1]
                            if r != 3:
                                subj_type, obj_type = target_types[0], target_types[1]
                                if ((e1_type == subj_type and e2_type == obj_type) or
                                    (e1_type == obj_type and e2_type == subj_type)):
                                    filtered.append((e1, e2))
                            else:
                                valid_locs = set(target_types[1:])
                                if ((e1_type == "PERSON" and e2_type in valid_locs) or
                                    (e2_type == "PERSON" and e1_type in valid_locs)):
                                    filtered.append((e1, e2))

                        if filtered:
                            prompt = f"{prompt_template} Here is the sentence: {sent}. If no such relation is found, return nothing."
                            try:
                                time.sleep(5)
                                response = get_gemini(prompt, "gemini-2.0-flash", 100, 0.2, 1, 32)
                            except Exception as e:
                                print(f"\tGemini API error: {e}. Continuing...")
                                continue
                            if ";" in response:
                                for line in response.strip().split("\n"):
                                    parts = line.split(";")
                                    if len(parts) != 2:
                                        continue
                                    subj, obj = parts[0].strip(), parts[1].strip()
                                    if 'unknown' in obj.lower() or 'student' in obj.lower():
                                        continue
                                    key = (subj, obj)
                                    if key not in X:
                                        print(f"\t✔ Gemini: {subj} → {obj}")
                                        X[key] = 1.0

                    elif method == 'spanbert':
                        for tokens, subj_ent, obj_ent in pairs:
                            subj_text, subj_type = subj_ent[0], subj_ent[1]
                            obj_text, obj_type = obj_ent[0], obj_ent[1]
                            marked = sent.text.replace(subj_text, f"[E1] {subj_text} [/E1]").replace(obj_text, f"[E2] {obj_text} [/E2]")
                            inputs = tokenizer(marked, return_tensors="pt", truncation=True, max_length=512)
                            with torch.no_grad():
                                outputs = model(**inputs)
                                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                                pred_idx = torch.argmax(probs).item()
                                pred_label = label_map[pred_idx]
                                confidence = probs[pred_idx].item()

                                if r == 3:
                                    valid_labels = {
                                        "per:countries_of_residence",
                                        "per:cities_of_residence",
                                        "per:stateorprovinces_of_residence"
                                    }
                                    is_valid = pred_label in valid_labels
                                else:
                                    target_labels = {
                                        1: "per:schools_attended",
                                        2: "per:employee_of",
                                        4: "org:top_members/employees"
                                    }
                                    is_valid = pred_label == target_labels[r]

                                if is_valid and confidence >= t:
                                    key = (subj_text, obj_text)
                                    if key not in X or confidence > X[key]:
                                        print(f"\t✔ SpanBERT: {subj_text} → {obj_text} (conf={confidence:.2f})")
                                        X[key] = confidence

                except Exception as e:
                    print(f"\tSkipped sentence due to error: {e}")
                    continue

            processed.add(url)

        if len(X) >= k:
            print(f"\n{k} tuples extracted, stopping iteration.")
            break

        candidates = [pair for pair in X if pair not in queried]
        if not candidates:
            print("ISE has stalled.")
            break

        y = candidates[0]
        queried.add(y)
        current_query = f"{y[0]} {y[1]}"
        iteration += 1

    print(f"\n============ FINAL EXTRACTED RELATIONS ({len(X)}) ============\n")
    for (subj, obj), conf in list(X.items())[:k]:
        print(f"Subject: {subj}\tObject: {obj}\tConfidence: {conf:.2f}")
    print(f"\nTotal ISE iterations: {iteration + 1}")

if __name__ == "__main__":
    main()
