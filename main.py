import requests
import sys
from bs4 import BeautifulSoup
import spacy
from googleapiclient.discovery import build
import google.generativeai as genai
from spacy_help import get_entities, create_entity_pairs
import time
#from spanbert import SpanBERT



# We expect 5 additional arguments:

    #  4) Method: -spanbert or -gemini
    #  5) Relation r in {1, 2, 3, 4}
    #  6) Threshold t between 0 and 1 (ignored if method == -gemini)
    #  7) Seed query q (a list of words in double quotes, e.g. "bill gates microsoft")
    #  8) Integer k > 0

# ======= HARDCODED KEYS ========
API_KEY = 'AIzaSyAYiEosxKFAa3cwpyN-Au3H7wRhZtAx8KY'
CX = 'd29acf7ff2d2f40a9'
GEMINI_API_KEY = 'AIzaSyBzTTV9qedmDGr-IjLhNYT9pzOAHpHOOIc'
genai.configure(api_key=GEMINI_API_KEY)

# ======= RELATION CONFIGS ========
RELATION_MAP = {
    1: ("Schools_Attended", ["PERSON", "ORGANIZATION"]),
    2: ("Work_For", ["PERSON", "ORGANIZATION"]),
    3: ("Live_In", ["PERSON", "LOCATION", "CITY", "COUNTRY", "STATE_OR_PROVINCE"]),
    4: ("Top_Member_Employees", ["ORGANIZATION", "PERSON"])
}

relation_prompts = {
         1:"Identify any found school attended relationships in the sentence below. "
         "Return only if the subject is a person and the object is a school or university name. The object must be a school or university that the person went to."
         "Return output in the format Person ; School on its own line. Include the full name of the school, including 'college' or 'university' "
         "For example, if the sentence was: Jeff Bezos graduated from Princeton, the response should be: Jeff Bezos ; Princeton University.",
        
         2:"Identify any found work-for relationships in the sentence below. \n "
         "Return only if the subject is a person and the object is a company name of which the person worked for. Do not return schools they attended, only places they work/worked."
         "Return output in the format Person ; Place of work on its own line. "
         "For example, if the sentence was: Alec Radford is currently working with OpenAi, the response should be Alec Radford ; OpenAi",

         3:"Identify any live-in relationships in the sentence below.\n "
         "Return only if the subject is a person and the object is a location (city or country)."
         "Return output in the format Person ; Location where they live on its own line. "
        "For example if the sentence was: Mariah Carey moved to New York City last year, the response should be Mariah Carey ; New York City",

         4:"Identify any top-member or high-ranking employee relationships in the sentence below."
         "Return only  if the subject is an ORGANIZATION and the object is a PERSON. The person must be a high-ranking employee of that company. "
         "Return them in the format: Organization ; Person — on its own line. "
         "For example if the sentence was: Jensen Huang is the CEO of Nvidia, the response should be Nvidia ; Jensen Huang",

     }

# ==== NEW IMPORTS for SpanBERT ====
import torch
#from transformers import AutoTokenizer, AutoModelForSequenceClassification


def google_search(query):
    service = build("customsearch", "v1", developerKey=API_KEY)
    res = service.cse().list(q=query, cx=CX, num=10).execute()
    results = res.get("items", [])
    return [item['link'] for item in results]



def fetch_page_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["style", "script", "head", "title", "table", "nav", "header"]):
            tag.decompose()
        raw_text = soup.get_text(" ", strip=True)
        if len(raw_text) > 10000:
               print(f"\tTrimming webpage content from {len(raw_text)} to 10000 characters")
               raw_text = raw_text[:10000]
        print(f"\tWebpage length (num characters): {len(raw_text)}")
        print("\tAnnotating the webpage using spaCy...")
        return raw_text[:10000]
    except Exception as e:
        print(f"\tUnable to fetch URL. Continuing...")
        return ""
    


def get_gemini(prompt, model_name, max_tokens,temperature,top_p,top_k):
    model = genai.GenerativeModel(model_name)
    generation_config = genai.types.GenerationConfig(
    max_output_tokens=max_tokens,
         temperature=temperature,
         top_p=top_p,
         top_k=top_k
    )
    time.sleep(5)
    response = model.generate_content(prompt, generation_config=generation_config)

    return response.text.strip() if response.text else "No response received"


def main():

    # Parse CLI arguments
    method = sys.argv[1]
    r = int(sys.argv[2])
    t = float(sys.argv[3])
    seed_query = sys.argv[4]
    k = int(sys.argv[5])

    if method not in ["spanbert", "gemini"]:
        sys.exit("Method must be 'spanbert' or 'gemini'")
    if r not in RELATION_MAP:
        sys.exit("Relation must be an integer 1–4.")
    if method == "spanbert" and not (0 <= t <= 1):
        sys.exit("Confidence threshold must be between 0 and 1.")
    if k <= 0:
        sys.exit("k must be > 0")

    relation, entity_types = RELATION_MAP[r]

    print("\n========== Parameters ==========")
    print(f"Client key     = {API_KEY}")
    print(f"Engine key     = {CX}")
    print(f"Gemini key     = {GEMINI_API_KEY}")
    print(f"Method         = {method}")
    print(f"Relation       = {relation}")
    print(f"Threshold      = {t}")
    print(f"Query          = {seed_query}")
    print(f"# of Tuples    = {k}")
    print("================================\n")

    print("Loading necessary libraries...")
    nlp = spacy.load("en_core_web_sm")
    if method == 'spanbert':
        bert = SpanBERT(pretrained_dir="./pretrained_spanbert")


    # ==== Initialize sets ====
    X, processed, queried = set(), set(), set()
    current_query = seed_query
    it = 0
    total_extracted = 0

    while len(X) < k:
        print(f"\n=========== Iteration: {it} - Query: {current_query} ===========")
        urls = google_search(current_query)
        if not urls:
            print("No results found.")
            break
        index = 1
        for url in urls:
            if url in processed:
                continue
            print(f"\nURL ({index}/{len(urls)}): {url}")
            text = fetch_page_text(url)
            print(f"\tWebpage length: {len(text)} characters")
            if not text:
                continue

            doc = nlp(text)
            sentences = list(doc.sents)
            print(f"\tExtracted {len(sentences)} sentences. Processing each one for correct entity pairings...")

            extracted_count =0
            annotated = 0
            for idx, sent in enumerate(sentences):
                entity_pairs = create_entity_pairs(sent, entity_types)
                if (idx + 1) % 5 == 0 or idx == len(sentences) - 1:
                    print(f"\n\tProcessed {idx + 1} / {len(sentences)} sentences")

                if not entity_pairs:
                    continue
                if method == "gemini":
                    filtered_pairs = []
                    for _, e1, e2 in entity_pairs:
                        e1_type = e1[1]
                        e2_type = e2[1]
                        if r !=3:
                            subj_type, obj_type = entity_types
                            if ((e1_type == subj_type and e2_type == obj_type) or (e1_type == obj_type and e2_type == subj_type)):
                                filtered_pairs.append((e1, e2))
                        else:
                            valid_locations = set(entity_types[1:])
                            if ((e1_type == "PERSON" and e2_type in valid_locations) or (e2_type == "PERSON" and e1_type in valid_locations)):
                                filtered_pairs.append((e1, e2))

                    if filtered_pairs:
                        prompt=f"{relation_prompts[r]} Here is the sentence: {sent} If no such relation is found, return nothing."
                        model_name = "gemini-2.0-flash"
                        max_tokens = 100
                        temperature = 0.2
                        top_p = 1
                        top_k = 32
                        try:                                 
                            response = get_gemini(prompt, model_name, max_tokens, temperature, top_p, top_k)
                        except Exception as e:
                            print(f"\tGemini API error. Continuing...")
                            continue
                       
                        if ";" in response:
                            lines = response.strip().split("\n")
                            for line in lines:
                                extracted = line.split(";")
                                if len(extracted) !=2:
                                    continue
                                subject = extracted[0].strip()
                                obj = extracted[1].strip()    

                                if 'unknown' in obj.lower() or 'student' in obj.lower():
                                    continue
                                print(f"\n=== Extracted Relation ===")
                                clean_sent = ''.join(char for char in str(sent) if ord(char) < 128)
                                print(f"\nSentence: {sent} ")
                                print(f"Subject: {subject} | Object: {obj}")
                                if ((subject,obj,1.0)) in X:
                                    print("Duplicate. Ignoring this.")
                                else:
                                    print(f"\nAdding to set of extracted relations")
                                    X.add((subject, obj, 1.0))
                                    extracted_count +=1
                                annotated +=1
                                print(f"==========")  

                elif method == "spanbert":
                    examples = []
                    for tokens, e1, e2 in entity_pairs:
                        e1_span = e1[2]
                        e2_span = e2[2]
                        example = {
                            "tokens": tokens,
                            "subj": (e1[0], e1[1], e1_span),
                            "obj": (e2[0], e2[1], e2_span)
                        }
                        examples.append(example)

                    predictions = bert.predict(examples)
                    for idx, (label, conf) in enumerate(predictions):
                        if r == 3:
                            live_in_labels = {
                                "per:countries_of_residence",
                                "per:cities_of_residence",
                                "per:stateorprovinces_of_residence"
                            }
                            valid_label = label in live_in_labels
                        else:
                            target_labels = {
                                1: "per:schools_attended",
                                2: "per:employee_of",
                                4: "org:top_members/employees"
                            }
                            valid_label = label == target_labels[r]

                        if valid_label and conf >= t:
                            subj = examples[idx]["subj"][0]
                            obj = examples[idx]["obj"][0]
                            if (subj, obj, conf) not in X:
                                print(f"\t✔ SpanBERT: {subj} → {obj} | Relation: {label} | Confidence: {conf:.2f}")
                                X.add((subj, obj, conf))

                    

            processed.add(url)
            index+=1
            print(f"Extracted annotations for {annotated} out of a total {len(sentences)} sentences for this website.")
            total_extracted += extracted_count
            print(f"Relations extracted from this website: {extracted_count}. Total: {total_extracted}")
                

        # Check stopping criteria
        if len(X) < k:
            candidates = [t for t in X if t not in queried]
            if not candidates:
                print("ISE has stalled.")
                break
            y = candidates[0]
            queried.add(y)
            current_query = f"{y[0]} {y[1]}"
            it += 1

    # Final output
    print(f"\n============ Final Extracted Relations ({len(X)}) ============")
    for subj, obj, conf in list(X)[:k]: #prints top k extracted relations
        print(f"Subject: {subj}\tObject: {obj}\tConfidence: {conf:.2f}")
    print(f"\nTotal number of iterations: {it}")


if __name__ == "__main__":
    main()

