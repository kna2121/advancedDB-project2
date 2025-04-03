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
GEMINI_API_KEY = 'AIzaSyBH4UBuhu9hLgk4mDUkQfbeZkxkPz4toNk'
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

# ======================= HELPERS ============================
def google_search(api_key, cx, query):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cx, num=10).execute()
    return [item['link'] for item in res.get("items", [])]

# Uses beautiful soup to get plain text from URL and ignore extraneous html tags
# Parameters: url
# Returns extracted text from webpage
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

#Uses gemimi api key to prompt using specified prompt gemini and return text response
def get_gemini(prompt, model_name, max_tokens, temperature, top_p, top_k):
    time.sleep(5)
    model = genai.GenerativeModel(model_name)
    config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    response = model.generate_content(prompt, generation_config=config)
    return response.text.strip() if response.text else "No response received"

# ======================= MAIN ============================
def main():

    #take in command line arguments
    method = sys.argv[1]
    r = int(sys.argv[2])
    t = float(sys.argv[3])
    seed_query = sys.argv[4]
    k = int(sys.argv[5])

    # make sure arguments are valid
    if method not in ['spanbert', 'gemini']:
        sys.exit("Method must be 'spanbert' or 'gemini'")
    if r not in relation_pairs:
        sys.exit("Relation must be an integer from 1 to 4.")
    if method == 'spanbert' and not (0 <= t <= 1):
        sys.exit("Threshold must be between 0 and 1.")
    if k <= 0:
        sys.exit("k must be > 0.")

#print out arguments
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

    print("Loading necessary libraries...")
    nlp = spacy.load("en_core_web_sm")

    #initialize spanbert model
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
    #initialize sets
    X = set()  
    processed, queried = set(), set()
    current_query = seed_query
    iteration = 0
    total_extracted = 0

    #continue until k tuples are found, check after each iteration of 10 urls
    while len(X) < k:
        print(f"\n=========== Iteration: {iteration} - Query: {current_query} ===========")
        urls = google_search(API_KEY, CX, current_query)
        if not urls:
            print("No search results found.")
            break
        
        index = 1
        for url in urls: #iterate through urls
            if url in processed:
                continue
            print(f"\nURL ({index}/{len(urls)}): {url}")
            index+=1
            print("\tFetching text from url...")
            raw_text = fetch_page_text(url)
            if not raw_text:
                continue

            doc = nlp(raw_text) #use spacy
            sentences = list(doc.sents)
            print(f"\tExtracted {len(sentences)} sentences. Processing each one for correct entity pairings...")

            extracted_count =0
            annotated = 0
            for i, sent in enumerate(sentences):
                try:
                    pairs = create_entity_pairs(sent, target_types) #check for entity pairs
                    if not pairs:
                        continue
                    if (i+1) % 5 == 0 or i == len(sentences) - 1:
                        print(f"\n\tProcessed {i + 1} / {len(sentences)} sentences")

                    #entity pairs must contain BOTH kinds of entities for the sentence to be prompted into gemini
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
                        #if sentence contains required entity types, prompt sentence into gemini
                        if filtered:
                            prompt = f"{prompt_template} Here is the sentence: {sent}. If no such relation is found, return nothing."
                            try:
                                response = get_gemini(prompt, "gemini-2.0-flash", 100, 0.2, 1, 32)
                            except Exception as e:
                                print(f"\tGemini API error: {e}. Continuing...")
                                continue
                            #get returned relation
                            if ";" in response:
                                lines = response.strip().split("\n")
                                for line in lines:
                                    extracted = line.split(";")
                                    if len(extracted) != 2:
                                        continue
                                    subj, obj = extracted[0].strip(), extracted[1].strip()
                                    if 'unknown' in obj.lower() or 'student' in obj.lower():
                                        continue
                                    print(f"\n=== Extracted Relation ===")
                                    clean_sent = ''.join(char for char in str(sent) if ord(char) < 128)
                                    print(f"\nSentence: {clean_sent} ")
                                    print(f"Subject: {subj} | Object: {obj}")
                                    if ((subj,obj,1.0)) in X:
                                        print("Duplicate. Ignoring this.")
                                    else:
                                        print(f"\nAdding to set of extracted relations")
                                        X.add((subj, obj, 1.0))
                                        extracted_count +=1
                                    annotated +=1
                                    print(f"==========")  

                    elif method == 'spanbert':
                        for _, subj_ent, obj_ent in pairs:
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
                                    if (subj_text, obj_text) not in X:
                                        X.add(subj_text, obj_text,confidence)
                                        print(f"\t Subject: {subj_text} | Object:{obj_text} (conf={confidence})")

                except Exception as e:
                    print(f"\tSkipped sentence due to error: {e}")
                    continue

            processed.add(url)
            
            if method == 'gemini':
                print(f"Extracted annotations for {annotated} out of a total {len(sentences)} sentences for this website.")
                total_extracted += extracted_count
                print(f"Relations extracted from this website: {extracted_count}. Total: {total_extracted}")
            

        
        if len(X) < k:    
            candidates = [tup for tup in X if tup not in queried]
            if not candidates:
                print("ISE has stalled.")
                break

            y = candidates[0]
            queried.add(y)
            current_query = f"{y[0]} {y[1]}"
            iteration += 1

    #prints out top-k extracted relations 
    print(f"\n============ FINAL EXTRACTED RELATIONS ({len(X)}) ============\n")
    for subj, obj, conf in list(X)[:k]: #prints top k extracted relations
        if method == 'spanbert':
            print(f"Subject: {subj}\tObject: {obj}\tConfidence: {conf:.2f}")
        elif method == 'gemini':
            print(f"Subject: {subj}\tObject: {obj}")
    print(f"\nTotal ISE iterations: {iteration + 1}")

if __name__ == "__main__":
    main()
