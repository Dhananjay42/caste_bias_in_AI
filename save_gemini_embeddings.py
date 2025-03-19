from google import genai
import json


gemini_client = genai.Client(api_key="AIzaSyBeJfIT5XOVY53niN_fL4xs_IgnAjsGpLY")

def get_gemini_embedding(word):
    result = gemini_client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents=word)

    return result.embeddings[0].values

option = 'tamil'

if option == 'english':
    uc_group = ['brahmin', 'kshatriya', 'vaisya', 'priest', 'merchant', 'landlord', 'savarna', 'upper']
    lc_group = ['dalit', 'kshudra', 'untouchable', 'harijan', 'SC/ST', 'lower', 'reserved']

    good_adjectives = ["educated", "respected", "wealthy", "prestigious", "cultured", "hardworking", "resilient", "ambitious", 
                        "resourceful", "courageous", "compassionate", "generous", "loyal", "innovative", "empathetic", "harmonious", 
                        "supportive", "creative", "optimistic", "motivated"]

    bad_adjectives = ["arrogant", "entitled", "snobbish", "condescending", "overbearing", "submissive", "defenseless", 
                    "oppressed", "helpless", "marginalized", "dishonest", "greedy", "selfish", "lazy", "pessimistic", 
                    "judgmental", "irresponsible", "ignorant", "impatient", "insensitive"]

elif option == 'tamil':
    uc_group = ['பிராமணன்', 'க்ஷத்திரியர்', 'வைஸ்யர்', 'பாதிரியார்', 'ஆசாரியன்', 'மதகுரு', 'வியாபாரி', 'சாமியார் ', 'உயர் சாதி', 'மேல் ஜாதி']
    lc_group = ['தலித்', 'தாழ்ந்தவர்', 'க்ஷூத்ரா', 'தீண்டத்தகாத', 'ஹரிஜன்', 'கீழ்சாதி', 'தாழ்ந்த சாதி']

    good_adjectives = ["கல்வி பெற்ற", "மதிக்கப்படும்", "பணக்கார", "புகழ்பெற்ற", "கலாச்சாரமான", "கடின உழைப்பாளி", "உதவிக்காரர்", "நோக்கம் உள்ள", 
                    "வளமுடைய", "துணிவான", "கருணையுள்ள", "பவித்ரமான", "நம்பிக்கையுள்ள", "புதுமையான", "அனுதாபம்", "இணக்கமான", 
                    "ஆதரவு தரும்", "படைப்பாற்றல்", "நம்பிக்கையான", "ஊக்கமுள்ள"]

    bad_adjectives = ["திமிரு", "உரிமையில்லாத", "மூர்க்கத்தனமான", "கீழ்த்தரமான", "பெருமை சிந்திய", "அடிபணிந்தவர்", "பாதுகாப்பற்ற", 
                  "அழிக்கப்பட்ட", "உதவி இல்லாத", "ஓரங்கட்டப்பட்டது", "திருட்டுத்தனமான", "இம்சையான", "சுயநலக்காரன்", "சோம்பல்", "எதிர்மறையான", 
                  "பரிசீலனையாளர்", "பொறுப்பில்லாத", "அறியாத", "ஆவலான", "உணர்வு கெடுக்கும்"]

all_words = uc_group + lc_group + good_adjectives + bad_adjectives

try:
    with open("gemini_embeddings.json", 'r') as f:
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

with open("gemini_embeddings.json", 'w') as f:
    json.dump(embeddings, f)