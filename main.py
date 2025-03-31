import requests
from bs4 import BeautifulSoup
import spacy
from googleapiclient.discovery import build

API_KEY = 'AIzaSyAYiEosxKFAa3cwpyN-Au3H7wRhZtAx8KY'

CX = 'd29acf7ff2d2f40a9'

def google_search(query):
    print(f"Parameters:\n Client key = {API_KEY}\n Engine Key = {CX}\n Gemini Key = {GEMINI_KEY}\n Method = gemini\n")
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
    query = "sergey brin stanford"
    relation = "Schools_Attended"

    print("Loading necessary libraries...\n")
    nlp = spacy.load("en_core_web_sm")

    urls = google_search(query)
    if not urls:
        print("No search results found.")
        return

    first_url = urls[0]
    print(f"\nURL (1 / {len(urls)}): {first_url}")
    print("\tFetching text from url ...")

    raw_text = fetch_page_text(first_url)
    if len(raw_text) > 10000:
        print(f"\tTrimming webpage content from {len(raw_text)} to 10000 characters")
        raw_text = raw_text[:10000]

    print(f"\tWebpage length (num characters): {len(raw_text)}")

    print("\tAnnotating the webpage using spaCy...")
    doc = nlp(raw_text)
    sentences = list(doc.sents)
    print(f"\tExtracted {len(sentences)} sentences.")
    print("\nChecking for entity pairs (PERSON + ORGANIZATION):")
    for sent in sentences:
        ents = [ent for ent in sent.ents]
        persons = [ent for ent in ents if ent.label_ == "PERSON"]
        orgs = [ent for ent in ents if ent.label_ == "ORG"]
        if persons and orgs:
            print(f"\nSentence: {sent.text.strip()}")
            print(f"  PERSON entities: {[p.text for p in persons]}")
            print(f"  ORG entities: {[o.text for o in orgs]}")

if __name__ == "__main__":
    main()
