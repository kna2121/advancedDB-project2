from googleapiclient.discovery import build

API_KEY = 'AIzaSyAYiEosxKFAa3cwpyN-Au3H7wRhZtAx8KY'
CX = 'd29acf7ff2d2f40a9'
gemini_key="AIzaSyBzTTV9qedmDGr-IjLhNYT9pzOAHpHOOIc"


# Performs a Google Custom Search for user's query using the API key and engine keys
# Params: The string to search/query
# Returns a list of tuples, each containing the title, URL, and snippet of a search result from top 10

def google_search(query):

    print(f"Parameters:\n Client key = \n Engine Key:{CX}\n Gemini Key:{gemini_key}\n Method=\n")


    # Initialize the service using the api key
    service = build("customsearch", "v1", developerKey=API_KEY)

    #  execute the query and get the top 10 results
    
    res = service.cse().list(q=query, cx=CX, num=10).execute()
    results = res.get("items", [])
    search_results = []

    # process each result and store as (title, link, snippet)
    for i, item in enumerate(results, 1):
        search_results.append((item['title'], item['link'], item.get('snippet', '')))
    
    return search_results