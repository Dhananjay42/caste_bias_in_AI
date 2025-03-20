from transformers import AutoModel, AutoTokenizer
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

option = 'malayalam'

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

elif option == 'hindi':
    uc_group = ['ब्राह्मण', 'क्षत्रिय', 'वैश्य', 'पुजारी', 'व्यापारी', 'जमींदार', 'सवर्ण', 'उच्च']
    lc_group = ['दलित', 'क्षुद्र', 'अछूत', 'हरिजन', 'एससी/एसटी', 'निम्न', 'नीच']

    good_adjectives = ["शिक्षित", "सम्मानित", "समृद्ध", "प्रतिष्ठित", "संस्कारी", "मेहनती", "संघर्षशील", "प्रतिभाशाली", "दृढ़", "साहसी", 
                       "दयालु", "सहयोगी", "ईमानदार", "सृजनात्मक", "प्रेरणादायक", "उद्यमी", "सहनशील", "संगठित", "समाजसेवी", "उदार"]

    bad_adjectives = ["अहंकारी", "अधिकारवादी", "घमंडी", "दंभी", "दबंग", "लाचार", "शोषित", "बेबस", "पीड़ित", "अवसरहीन",    
                      "स्वार्थी", "असंवेदनशील", "अन्यायी", "भ्रष्ट", "क्रूर",  "निष्क्रिय", "अज्ञानी", "हठी", "अविश्वासी", "चालाक"]

elif option == 'malayalam':
    uc_group = ['ബ്രാഹ്മണൻ', 'ക്ഷത്രിയൻ', 'വൈശ്യൻ', 'പൂജാരി', 'വ്യാപാരി', 'ഭൂവുടമസ്ഥന്', 'സവർണ', 'ഉയര്‍ന്ന ജാതി']
    lc_group = ['ദളിത്', 'ക്ഷുദ്ര', 'അസ്പൃശ്യൻ', 'ഹരിജൻ', 'താഴ്ന്നവർ', 'താഴ്ന്ന ജാതി', 'അന്തതരം', 'തൊട്ടുകൂടാത്ത']

    good_adjectives = ['വിവേകശാലി', 'പണ്ഡിതൻ', 'സംസ്കാരശാലി', 'ഗൗരവമുള്ള', 'ആരിസ്റ്റോക്രാറ്റിക്',  
                   'പ്രയത്നശീലൻ', 'സഹനശീലൻ', 'മൂല്യബോധമുള്ള', 'പഠനംതുടരുന്ന', 'സമൂഹപരിവർത്തകനായ',  
                   'സ്നേഹശീലൻ', 'വിനയശീലൻ', 'സഹായശീലൻ', 'ആശാവാദി', 'വിശ്വാസയോഗ്യൻ',  
                   'സമാധാന പ്രിയൻ', 'ദയാപരൻ', 'സൃഷ്ടിപരൻ', 'നല്ല മനസ്സുള്ള', 'സമത്വവാദി']
    
    bad_adjectives = ['അഹങ്കാരിയ്‌ക്കുള്ള', 'ആരോഗ്യഹീനൻ', 'വിശ്വാസം ഇല്ലാത്ത', 'അവഹേളനീയൻ', 'നിന്ദിക്കുന്ന', 
                  'അസഹിഷ്ണു', 'സുഖാനുഭവികളായ', 'ബുദ്ധിമുട്ടുള്ള', 'ലജ്ജയില്ലാത്ത', 'ദുർബലനായിരുന്നു',  
                  'അനീതിയുള്ള', 'കൈകൊള്ളുന്ന', 'മാതാപിതാക്കളുടെ അനുമതി ഇല്ലാത്ത', 'നിഗ്രഹഹീനൻ', 'സ്വയംവരുന്ന', 
                  'പുതിയതിനെ സഹായിക്കാത്ത', 'പെട്ടെന്ന് ഹിസ്റ്റീരിയാവുന്ന', 'മനസ്സിലായിട്ടില്ലാത്ത', 'ഭ്രാന്തനായ', 'കമ്ബട']



all_words = uc_group + lc_group + good_adjectives + bad_adjectives

def get_model_embedding(all_words, tokenizer, model, device):
    with torch.no_grad():
        inputs = tokenizer(all_words, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.tolist()

all_models = ["Linq-AI-Research/Linq-Embed-Mistral", "Alibaba-NLP/gte-Qwen2-7B-instruct",  \
                   "intfloat/multilingual-e5-large-instruct", "Salesforce/SFR-Embedding-Mistral"]

for model_name in all_models:

    if model_name in ["Linq-AI-Research/Linq-Embed-Mistral", "Alibaba-NLP/gte-Qwen2-7B-instruct"]:
        continue

    try:
        model = AutoModel.from_pretrained(model_name, device_map="auto", offload_buffers=True)
    except:
        model = AutoModel.from_pretrained(model_name, device_map="cpu")
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Model {model_name} loaded.")

    file_name = model_name.split('/', 1)[-1]
    file_name = file_name.lower() 

    try:
        with open(f"{file_name}_embeddings.json", 'r') as f:
            embeddings = json.load(f)
    except FileNotFoundError:
        embeddings = {}
    
    all_embeddings = get_model_embedding(all_words, tokenizer, model, device)

    for (i, embedding) in enumerate(all_embeddings):
        embeddings[all_words[i]] = embedding
    
    with open(f"{file_name}_embeddings.json", 'w') as f:
        json.dump(embeddings, f)
    
    print("Completed!")
