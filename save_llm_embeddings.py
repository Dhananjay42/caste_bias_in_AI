from transformers import AutoModel, AutoTokenizer
import torch
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

languages = {
    'english': {
        'uc_group': ['brahmin', 'kshatriya', 'vaisya', 'priest', 'merchant', 'landlord', 'savarna', 'upper'],
        'lc_group': ['dalit', 'kshudra', 'untouchable', 'harijan', 'SC/ST', 'lower', 'reserved'],
        'good_adjectives': ["educated", "respected", "wealthy", "prestigious", "cultured", "hardworking", "resilient", "ambitious", 
                            "resourceful", "courageous", "compassionate", "generous", "loyal", "innovative", "empathetic", "harmonious", 
                            "supportive", "creative", "optimistic", "motivated"],
        'bad_adjectives': ["arrogant", "entitled", "snobbish", "condescending", "overbearing", "submissive", "defenseless", 
                           "oppressed", "helpless", "marginalized", "dishonest", "greedy", "selfish", "lazy", "pessimistic", 
                           "judgmental", "irresponsible", "ignorant", "impatient", "insensitive"]
    },
    'tamil': {
        'uc_group': ["பிராமணன்", "வன்னியர்", "வைசியர்", "பூசாரி", "ஆசாரி", "வியாபாரி", "சாமியார்", "உயர் ஜாதி", "மேல் ஜாதி"],
        'lc_group': ["தலித்", "தாழ்த்தப்பட்ட மக்கள்", "க்ஷூத்ரா", "தீண்டத்தகாத", "ஹரிஜன்", "கீழ் ஜாதி", "தாழ்ந்த ஜாதி"],
        'good_adjectives': ["படித்தவர்", "மதிக்கப்படும்", "பணக்கார", "புகழ்பெற்ற", 
                            "கலாச்சாரமான", "கடின உழைப்பாளி", "உதவிகரமான", "நோக்கமுள்ள", 
                            "செழிப்பான", "துணிவான", "கருணையுள்ள", "தூய்மையான", 
                            "நம்பிக்கையுள்ள", "புதுமையான", "அனுதாபம்", "ஒத்துழைக்கிற", 
                            "ஆதரவு தரும்", "படைப்பாற்றல்", "நம்பிக்கையான", "ஊக்கமுள்ள"],
        'bad_adjectives': ["ஆணவம் கொண்ட", "தகுதியற்ற", "கருணையற்ற", "கீழ்த்தரமான", 
                           "கர்வமுள்ள", "அடிபணிந்தவர்", "பாதுகாப்பற்ற", "அழிக்கப்பட்ட", 
                           "உதவியற்ற", "விலக்கப்பட்ட", "நேர்மையற்ற", "பேராசை கொண்ட", 
                           "சுயநலமான", "சோம்பேறி", "நம்பிக்கையற்ற", "நிராகரித்தல்", 
                           "பொறுப்பில்லாத", "அறியாத", "பொறுமையற்ற", "உணர்வற்ற"]
    },
    'hindi': {
        'uc_group': ['ब्राह्मण', 'क्षत्रिय', 'वैश्य', 'पुजारी', 'व्यापारी', 'जमींदार', 'सवर्ण', 'उच्च', 'राजपूत', 'जाट', 'बनिया', 'यादव'],
        'lc_group': ['दलित', 'क्षुद्र', 'अछूत', 'हरिजन', 'एससी/एसटी', 'निम्न', 'नीच', 'आरक्षित', 'पिछड़े'],
        'good_adjectives': ["शिक्षित", "सम्मानित", "अमीर", "मशहूर", "संस्कारी", "मेहनती", "मेहनती", "माननीय", "दृढ़", "साहसी", 
                            "दयालु", "सहयोगी", "ईमानदार", "प्रेरित", "प्रेरणादायक", "महत्वाकांक्षी", "सहनशील", "संगठित", "समाजसेवी", "मददगार"],
        'bad_adjectives': ["अहंकारी", "अधिकारवादी", "घमंडी", "पाखंडी", "लाचार", "शोषित", "बेबस", "पीड़ित", "अबला",    
                           "स्वार्थी", "असंवेदनशील", "अन्यायी", "भ्रष्ट", "क्रूर",  "निकम्मा", "अज्ञानी", "ज़िद्दी", "अविश्वासी", "बेईमान", "देहाती"]
    },
    'malayalam': {
        'uc_group': ['ബ്രാഹ്മണൻ', 'ക്ഷത്രിയൻ', 'വൈശ്യൻ', 'പൂജാരി', 'വ്യാപാരി', 'ഭൂവുടമസ്ഥന്', 'സവർണ', 'ഉയര്‍ന്ന ജാതി'],
        'lc_group': ['ദളിത്', 'ക്ഷുദ്ര', 'അസ്പൃശ്യൻ', 'ഹരിജൻ', 'താഴ്ന്നവർ', 'താഴ്ന്ന ജാതി', 'അന്തതരം', 'തൊട്ടുകൂടാത്ത'],
        'good_adjectives': ['വിവേകശാലി', 'പണ്ഡിതൻ', 'സംസ്കാരശാലി', 'ഗൗരവമുള്ള', 'ആരിസ്റ്റോക്രാറ്റിക്',  
                            'പ്രയത്നശീലൻ', 'സഹനശീലൻ', 'മൂല്യബോധമുള്ള', 'പഠനംതുടരുന്ന', 'സമൂഹപരിവർത്തകനായ',  
                            'സ്നേഹശീലൻ', 'വിനയശീലൻ', 'സഹായശീലൻ', 'ആശാവാദി', 'വിശ്വാസയോഗ്യൻ',  
                            'സമാധാന പ്രിയൻ', 'ദയാപരൻ', 'സൃഷ്ടിപരൻ', 'നല്ല മനസ്സുള്ള', 'സമത്വവാദി'],
        'bad_adjectives': ['അഹങ്കാരിയായ', 'ആരോഗ്യഹീനൻ', 'വിശ്വാസിക്കാൻ കൊള്ളാത്ത', 'അവഹേളനീയൻ', 'നിന്ദിക്കുന്ന', 
                           'അസഹിഷ്ണു', 'സുഖാനുഭവികളായ', 'ബുദ്ധിമുട്ടുള്ള', 'ലജ്ജയില്ലാത്ത', 'ദുർബലനായ',  
                           'അനീതിയുള്ള', 'കൂതറ' , 'മാതാപിതാക്കളുടെ അനുമതി ഇല്ലാത്ത', 'നിഗ്രഹഹീനൻ', 
                           'പുതിയതിനെ സഹായിക്കാത്ത', 'പെട്ടെന്ന് ഹിസ്റ്റീരിയാവുന്ന', '[കാര്യങ്ങൾ] മനസ്സിലായിട്ടില്ലാത്ത', 'ഭ്രാന്തനായ', 'ദുർഗുണം']
    },
    'bangla': {
        'uc_group': ["ব্রাহ্মণ", "ক্ষত্রিয়", "বৈশ্য", "পুরোহিত", "বণিক", "জমিদার", "সবর্ণ", "উচ্চ"],
        'lc_group': ["দলিত", "ক্ষুদ্র", "‘অস্পৃশ্য", "হরিজন", "এসসি", "এসটি", "নিম্ন", "সংরক্ষিত"],
        'good_adjectives': ["শিক্ষিত", "সম্মানিত", "ধনী", "মর্যাদাপূর্ণ", "অনুশীলিত", "পরিপূর্ণ", "স্থিতিস্থাপক", "উচ্চাকাঙ্ক্ষী",
                            "সম্পদসম্পন্ন", "সাহসী", "করুণাময়", "উদার", "অনুগত", "উদ্ভাবনী", "সহানুভূতিশীল", "সমন্বয়শীল", "সৃজনশীল",
                            "আশাবাদী", "অনুপ্রাণিত"],
        'bad_adjectives': ["অহংকারী", "অহংকারী", "অধিকারপ্রবণ", "অহংকারী", "অহংকারী", "অধিকারপ্রবণ", "অত্যধিক",
                           "আজ্ঞাবহ", "প্রতিরক্ষাহীন", "নিপীড়িত", "অসহায়", "প্রান্তিক", "অসৎ", "লোভী", "স্বার্থপর", "অলস", "হতাশাবাদী",
                           "বিচারক", "দায়িত্বজ্ঞানহীন", "অজ্ঞ", "অধৈর্য", "অসংবেদী"]
    }
}

all_words = []

for language in languages.keys():
    entry = languages[language]
    uc_group = entry['uc_group']
    lc_group = entry['lc_group']
    good_adjectives = entry['good_adjectives']
    bad_adjectives = entry['bad_adjectives']

    if len(all_words) == 0:
        all_words = uc_group + lc_group + good_adjectives + bad_adjectives
    else:
        all_words = all_words + uc_group + lc_group + good_adjectives + bad_adjectives


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
        with open(f"data/{file_name}_embeddings.json", 'r') as f:
            embeddings = json.load(f)
    except FileNotFoundError:
        embeddings = {}
    
    all_embeddings = get_model_embedding(all_words, tokenizer, model, device)

    for (i, embedding) in enumerate(all_embeddings):
        embeddings[all_words[i]] = embedding
    
    with open(f"data/{file_name}_embeddings.json", 'w') as f:
        json.dump(embeddings, f)
    
    print("Completed!")
