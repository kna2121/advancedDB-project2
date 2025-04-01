import requests
import sys
from bs4 import BeautifulSoup
import spacy
from googleapiclient.discovery import build


# We expect 5 additional arguments:

    #  4) Method: -spanbert or -gemini
    #  5) Relation r in {1, 2, 3, 4}
    #  6) Threshold t between 0 and 1 (ignored if method == -gemini)
    #  7) Seed query q (a list of words in double quotes, e.g. "bill gates microsoft")
    #  8) Integer k > 0

def google_search(API_KEY,CX, query):
    print(f"=========== Iteration: 0 - Query: {query} ===========")

    service = build("customsearch", "v1", developerKey=API_KEY)
    res = service.cse().list(q=query, cx=CX, num=10).execute()
    results = res.get("items", [])
    return [item['link'] for item in results]

def fetch_page_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    except Exception as e:
        print(f"\t[!] Failed to fetch {url}: {e}")
        return ""

def main():

    API_KEY = 'AIzaSyAYiEosxKFAa3cwpyN-Au3H7wRhZtAx8KY'
    CX = 'd29acf7ff2d2f40a9'
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


    X = set() #initialize set
    processed = set()

    urls = google_search(API_KEY, CX, seed_query) #query google search engine 
    if not urls:
        print("No search results found.")
        return

    urls = urls[:1]
    index = 1 
    for url in urls:
        if url not in processed:
            print(f"\nURL ({index} / {len(urls)}): {url}")
            print("\tFetching text from url ...")

            raw_text = fetch_page_text(url)

            if len(raw_text) > 10000:
                print(f"\tTrimming webpage content from {len(raw_text)} to 10000 characters")
                raw_text = raw_text[:10000]

            print(f"\tWebpage length (num characters): {len(raw_text)}")

            print("\tAnnotating the webpage using spaCy...")
            doc = nlp(raw_text) #use spacy
            sentences = list(doc.sents) # get entities
            
            print(f"\tExtracted {len(sentences)} sentences.")
            index+=1


if __name__ == "__main__":
    main()
