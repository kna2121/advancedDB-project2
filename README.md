a.
- Kira Ariyan - kna2121  
- Ralph Betesh - rb3557  

b. Submitted Files
- integrated.py  
- spacy_help.py  
- spanbert.py  
- example_relations.py  
- pretrained_spanbert/ (folder with model + config files)  
- requirements.txt  
- README.md
  
c.

sudo apt update
sudo apt install -y python3 python3-pip python3.8-venv build-essential


python3 -m venv dbproj
source dbproj/bin/activate


pip install -r requirements.txt
python -m spacy download en_core_web_sm


python integrated.py [method] [relation_id] [threshold] [seed_query] [k]

ex:
python integrated.py spanbert 2 0.7 "bill gates microsoft" 10

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

1. Perform Google search using the current query  
2. Scrape and clean page text  
3. Split into sentences w spaCy  
4. Extract valid (subject, object) entity pairs  
5. Use:
   - SpanBERT - predict relations and filter by confidence  
   - Gemini - prompt based relation extraction  
6. Add valid pairs to result set  
7. Use new pair to construct next query  
8. Repeat until k tuples extracted or no progress  

f.
  API_KEY = 'AIzaSyAYiEosxKFAa3cwpyN-Au3H7wRhZtAx8KY'
  CX = 'd29acf7ff2d2f40a9'
g.

- Both spanbert and gemini methods supported from CLI  
- We hardcoded API keys used in script  
- Model inference and prompt timing include 5s delay to prevent rate limits 429 errors  
