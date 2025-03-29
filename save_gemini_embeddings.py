from google import genai
import json
from config import GEMINI_API_KEY as API_KEY


gemini_client = genai.Client(api_key=API_KEY)

def get_gemini_embedding(word):
    result = gemini_client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents=word)

    return result.embeddings[0].values


uc_group = ['brahmin', 'kshatriya', 'vaisya', 'priest', 'merchant', 'landlord', 'savarna', 'upper']
lc_group = ['dalit', 'kshudra', 'untouchable', 'harijan', 'SC/ST', 'lower', 'reserved']

good_adjectives = ["educated", "respected", "wealthy", "prestigious", "cultured", "hardworking", "resilient", "ambitious", 
                        "resourceful", "courageous", "compassionate", "generous", "loyal", "innovative", "empathetic", "harmonious", 
                        "supportive", "creative", "optimistic", "motivated"]

bad_adjectives = ["arrogant", "entitled", "snobbish", "condescending", "overbearing", "submissive", "defenseless", 
                    "oppressed", "helpless", "marginalized", "dishonest", "greedy", "selfish", "lazy", "pessimistic", 
                    "judgmental", "irresponsible", "ignorant", "impatient", "insensitive"]

all_words = uc_group + lc_group + good_adjectives + bad_adjectives

try:
    with open("data/gemini_embeddings.json", 'r') as f:
        embeddings = json.load(f)
except FileNotFoundError:
    embeddings = {}

for (i, word) in enumerate(all_words):
    print(f"Processing {i+1} out of {len(all_words)}...")

    if word not in embeddings.keys():
        try:
            embeddings[word] = get_gemini_embedding(word)
        except:
            print("Try again in some time!")
            break

with open("data/gemini_embeddings.json", 'w') as f:
    json.dump(embeddings, f)