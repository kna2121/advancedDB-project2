import requests
import sys
from bs4 import BeautifulSoup
import spacy
from googleapiclient.discovery import build
import google.generativeai as genai
from spacy_help import get_entities, create_entity_pairs
import time




# We expect 5 additional arguments:

    #  4) Method: -spanbert or -gemini
    #  5) Relation r in {1, 2, 3, 4}
    #  6) Threshold t between 0 and 1 (ignored if method == -gemini)
    #  7) Seed query q (a list of words in double quotes, e.g. "bill gates microsoft")
    #  8) Integer k > 0

def google_search(API_KEY,CX, query):

    service = build("customsearch", "v1", developerKey=API_KEY)
    res = service.cse().list(q=query, cx=CX, num=10).execute()
    results = res.get("items", [])
    return [item['link'] for item in results]



def fetch_page_text(url):
    try:
        response = requests.get(url, timeout=10)
        content = response.text
        soup = BeautifulSoup(content, 'html.parser')
        for tag in soup(['table','nav','style','header']):
            tag.decompose()
            
       
        raw_text = soup.get_text(' ',strip=True)
        if len(raw_text) > 10000:
                print(f"\tTrimming webpage content from {len(raw_text)} to 10000 characters")
                raw_text = raw_text[:10000]
        print(f"\tWebpage length (num characters): {len(raw_text)}")
        print("\tAnnotating the webpage using spaCy...")
       
        return raw_text
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

    API_KEY = 'AIzaSyAYiEosxKFAa3cwpyN-Au3H7wRhZtAx8KY'
    CX = 'd29acf7ff2d2f40a9'
    GEMINI_API_KEY = 'AIzaSyBzTTV9qedmDGr-IjLhNYT9pzOAHpHOOIc'  # Substitute your own key here
    genai.configure(api_key=GEMINI_API_KEY)

    method = sys.argv[1]
    r = int(sys.argv[2])     
    t = float(sys.argv[3])   
    seed_query = sys.argv[4]  
    k = int(sys.argv[5]) 

    relations = ['0','Schools_Attended', 'Work_For', 'Live_In','Top_Member_Employees']
    relation = relations[r]

    if method not in ['spanbert', 'gemini']:
        sys.exit("Error: Please input valid method")
    
    if r < 1 or r > 4:
        sys.exit("Error: Please input an integer from 1 to 4 for r. ")
    
    if method == 'spanbert':
       if t<0 or t>1:
        sys.exit("Error: Please input a number between 0 and 1 for t. ")
    
    if k <=0:
        sys.exit("Error: Please input a number greater than 0 for k.")
    
    print(f"Parameters:\nClient key = {API_KEY}\nEngine Key = {CX}\nGemini Key = {'AIzaSyBzTTV9qedmDGr-IjLhNYT9pzOAHpHOOIc'}\nMethod = {method}\n")
    print(f"Relation\t= {relation}\nThreshold\t= {t}\nQuery    \t= {seed_query}\n# of Tuples     = {k}")

    print("Loading necessary libraries...\n")

    nlp = spacy.load("en_core_web_sm")

    relation_pairs = {
                1: ["PERSON", "ORGANIZATION"],  # Schools_Attended
                2: ["PERSON", "ORGANIZATION"],  # Work_For
                3: ["PERSON", "LOCATION", "CITY", "COUNTRY", "STATE_OR_PROVINCE"],      # Live_In
                4: ["ORGANIZATION", "PERSON"]   # Top_Member_Employees
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

    target_types = relation_pairs[r]

    #initialize sets
    X = set() 
    processed = set()
    queried = set()
    current_q =seed_query


    it = 0
    total_extracted=0
    while len(X) < k:
        print(f"=========== Iteration: {it} - Query: {current_q} ===========")
        urls = google_search(API_KEY, CX, current_q) #query google search engine 
        if not urls:
            print("No search results found.")
            return

        index = 1 
        for url in urls:
            if url not in processed:
                print(f"\nURL ({index} / {len(urls)}): {url}")
                print("\tFetching text from url ...")
                raw_text = fetch_page_text(url)

                if not raw_text:
                    continue

                doc = nlp(raw_text) #use spacy
                sentences = list(doc.sents)
                print(f"\tExtracted {len(sentences)} sentences. Processing each one for correct entity pairings...")

                
                extracted_count = 0 
                annotated=0
                for i, sent in enumerate(sentences):
                    try:
                        pairs = create_entity_pairs(sent, target_types)
                        if not pairs:
                            continue
                        filtered_pairs = []
                        for _, e1, e2 in pairs:
                            e1_type = e1[1]
                            e2_type = e2[1]
                            if r !=3:
                                subj_type, obj_type = target_types
                                if ((e1_type == subj_type and e2_type == obj_type) or (e1_type == obj_type and e2_type == subj_type)):
                                    filtered_pairs.append((e1, e2))
                            else:
                                valid_locations = set(target_types[1:])
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
                                response_text = get_gemini(prompt, model_name, max_tokens, temperature, top_p, top_k)
                            except Exception as e:
                                print(f"\tGemini API error. Continuing...")
                                continue
                            if (i + 1) % 5 == 0 or i == len(sentences) - 1:
                                print(f"\n\tProcessed {i + 1} / {len(sentences)} sentences")

                            if ";" in response_text:
                                lines = response_text.strip().split("\n")
                                for line in lines:
                                    extracted = line.split(";")
                                    if len(extracted) !=2:
                                        continue
                                    subject = extracted[0].strip()
                                    obj = extracted[1].strip()    

                                    if 'unknown' in obj.lower() or 'student' in obj.lower():
                                        continue

                                    print(f"\n=== Extracted Relation ===")
                                    sent = ''.join(char for char in str(sent) if ord(char) < 128)

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
                    except Exception as e:
                        print(f"Skipped sentence due to error. Continuing...")
                        continue
                                
                            
                print(f"Extracted annotations for {annotated} out of a total {len(sentences)} for this website.")
                total_extracted += extracted_count
                print(f"Relations extracted from this website: {extracted_count}. Total: {total_extracted}")
                
                index+=1
                
                processed.add(url)
        if len(X) < k:
            candidates = [tup for tup in X if tup not in queried]
            if not candidates:
                print("ISE has stalled.")
                break
            y = candidates[0]
            queried.add(y)
            current_q = f"{y[0]} {y[1]}"
            it+=1
        else:
            print(f"{k} tuples extracted, stopping iteration")
            break
            
    
    print(f'\n============ALL EXTRACTED RELATIONS ({total_extracted})============')
    results = list(X)
    #top_k = results[:k]
    for tup in results:
        print(f'Subject: {tup[0]}\tObject: {tup[1]}')
    print(f"Total number of iterations:{it+1}")
       

if __name__ == "__main__":
    main()
