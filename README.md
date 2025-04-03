a.
- Kira Ariyan - kna2121  
- Ralph Betesh - rb3557  

b. Submitted Files
- project2.py 
- spacy_help.py  
- spanbert.py  
- example_relations.py  
- pretrained_spanbert/ (folder with model + config files)  
- requirements.txt  
- README.md
  
c.


sudo apt update
sudo apt install -y python3 python3-pip python3.10-venv build-essential

python3 -m venv dbproj
source dbproj/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm


python integrated.py [method] [relation_id] [threshold] [seed_query] [k]

ex:
python integratedCode.py spanbert 2 0.7 "bill gates microsoft" 10

- method: spanbert or gemini 
- relation_id: 1 = Schools_Attended, 2 = Work_For, 3 = Live_In, 4 = Top_Member_Employees  
- threshold: Confidence threshold (used for spanbert only)  
- seed_query: Query string  
- k: Number of tuples to extract 

d. 

- integrated.py: main pipeline  
- spacy_help.py: Entity extraction and pairing w spaCy  
- spanbert.py: Loads model  
- example_relations.py: Defines prompts and valid entities  
- pretrained_spanbert/: Contains model fikes (.bin, config.json, etc.)


used transformers, torch, spacy, googleapiclient, google.generativeai, bs4, requests, collections, itertools

e.

1. Take in command line arguments and validate each one, ensuring it is a proper option (spanbert/gemini) or a valid number within range
2. Perform Google search using the current query and google api key.
3. Scrape and clean page text using beautiful soup. Only extract relevant text we tried to eliminate as much of the extraneous text as possible, to not waste space within the 10000 characters on header or navigation text.
4. Split into sentences w spaCy.
5. Loop through each sentence, and examine entity pairings from each sentence. Check for  valid (subject, object) entity pairs.
Used create_entity_pairs from spacy_help.py to extract all entity pairings. 
If valid entity pairs were found in that sentence (based on which relation we are looking for) :
6. Use:
   - SpanBERT - predict relations and filter by confidence  
   - Gemini - prompt based relation extraction. If proper relations are found, prompt sentence and prompt into gemini to look for relations. Prompted gemini to return relations in a specific format, so that they could be easily extracted in the next step 
   if such a relation was found.
7. If relation is extracted, (and not already in set) add to set X and display proper information.
8. After one iteration is complete, if k tuples not met use new pair to construct next query  
9. Repeat until k tuples extracted or no further tuples exist to use as a query, in this case ISE has stalled..
10. Print top k tuples as result.

f.
  API_KEY = 'AIzaSyAYiEosxKFAa3cwpyN-Au3H7wRhZtAx8KY'
  CX = 'd29acf7ff2d2f40a9'
g.

- Both spanbert and gemini methods supported from CLI  
- We hardcoded API keys used in script, as to reduce the amount of command line arguments needed (as encouraged on Ed discussion).
- Model inference and prompt timing include 5s delay to prevent rate limits 429 errors  
