from transformers import BertModel
import torch
from transformers import BertTokenizer, BertModel
from transformers import AutoModel, AutoTokenizer
import numpy as np
import asyncio
import json
from sklearn.metrics.pairwise import cosine_similarity
try:
    asyncio.set_event_loop(asyncio.new_event_loop())
except RuntimeError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def load_indic_bert():
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert', use_fast=False)
    model = AutoModel.from_pretrained('ai4bharat/indic-bert')
    model.to(device)

    return model, tokenizer

def load_muril():
    path = 'google/muril-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path,output_hidden_states=True)
    model.to(device)

    return model, tokenizer


def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Forward pass through BERT model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract embeddings (use last hidden state)
    embeddings = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
    
    # Get mean pooling of the token embeddings
    sentence_embedding = embeddings.mean(dim=1)  # Shape: (batch_size, hidden_size)
    
    return sentence_embedding

def get_group_embeddings(group, model, tokenizer):
    """ Precompute embeddings for all words in group """
    group_embeddings = {}
    for word in group:
        group_embeddings[word] = get_embedding(word, model, tokenizer)
    return group_embeddings

def get_weat_score(word_embedding, groupA_embeddings, groupB_embeddings):
    """ Compute WEAT score by comparing word with precomputed embeddings """
    a_score = np.mean([torch.cosine_similarity(groupA_embeddings[wordA], word_embedding, dim=1).item() for wordA in groupA_embeddings])
    b_score = np.mean([torch.cosine_similarity(groupB_embeddings[wordB], word_embedding, dim=1).item() for wordB in groupB_embeddings])
    
    return a_score - b_score

def compute_weat_scores(good_adjectives, bad_adjectives, uc_group, lc_group, model, tokenizer):

    uc_group_embeddings = get_group_embeddings(uc_group, model, tokenizer)
    lc_group_embeddings = get_group_embeddings(lc_group, model, tokenizer)

    good_scores = [get_weat_score(get_embedding(word, model, tokenizer), uc_group_embeddings, lc_group_embeddings) for word in good_adjectives]
    bad_scores = [get_weat_score(get_embedding(word, model, tokenizer), uc_group_embeddings, lc_group_embeddings) for word in bad_adjectives]

    # Compute WEAT score for good adjectives
    good_score = np.sum(good_scores)
    bad_score = np.sum(bad_scores)
    stddev = np.std(good_scores + bad_scores)

    weat_score = (np.mean(good_scores) - np.mean(bad_scores))/(stddev)

    return weat_score

def bert_pipeline(good_adjectives, bad_adjectives, uc_group, lc_group):
    bert_model, bert_tokenizer = load_indic_bert()
    weat_score = compute_weat_scores(good_adjectives, bad_adjectives, uc_group, lc_group, bert_model, bert_tokenizer)
    return weat_score

def muril_pipeline(good_adjectives, bad_adjectives, uc_group, lc_group):
    muril_model, muril_tokenizer = load_muril()
    weat_score = compute_weat_scores(good_adjectives, bad_adjectives, uc_group, lc_group, muril_model, muril_tokenizer)
    return weat_score

def get_json_embedding(word, json_path):
    with open(json_path, 'r') as f:
        embeddings = json.load(f)
    
    return torch.tensor(embeddings[word]).unsqueeze(0)

def json_pipeline(good_adjectives, bad_adjectives, uc_group, lc_group, file_name):

    uc_group_embeddings = {word: get_json_embedding(word, file_name) for word in uc_group}
    lc_group_embeddings = {word: get_json_embedding(word, file_name) for word in lc_group}


    good_scores = [get_weat_score(get_json_embedding(word, file_name), uc_group_embeddings, lc_group_embeddings) for word in good_adjectives]
    bad_scores = [get_weat_score(get_json_embedding(word, file_name), uc_group_embeddings, lc_group_embeddings) for word in bad_adjectives]

    # Compute WEAT score for good adjectives
    good_score = np.sum(good_scores)
    bad_score = np.sum(bad_scores)
    stddev = np.std(good_scores + bad_scores)

    weat_score = (np.mean(good_scores) - np.mean(bad_scores))/(stddev)

    return weat_score

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


models = [
    ("indic-bert", bert_pipeline, None),
    ("muril", muril_pipeline, None),
    #("gemini-embedding-exp-03-07", json_pipeline, "data/gemini_embeddings.json"),
    ("linq-embed-mistral", json_pipeline, "data/linq-embed-mistral_embeddings.json"),
    ("gte-qwen2-7b-instruct", json_pipeline, "data/gte-qwen2-7b-instruct_embeddings.json"),
    ("multilingual-e5-large-instruct", json_pipeline, "data/multilingual-e5-large-instruct_embeddings.json"),
    ("sfr-embedding-mistral", json_pipeline, "data/sfr-embedding-mistral_embeddings.json"),
]



for language, entry in zip(languages.keys(), languages.values()):

    good_adjectives = entry['good_adjectives']
    bad_adjectives = entry['bad_adjectives']
    uc_group = entry['uc_group']
    lc_group = entry['lc_group']

    print(f"Results for {language} language:")

    for model_name, pipeline_func, embeddings_file in models:
    
        # Call the corresponding pipeline function
        if embeddings_file:
            score = pipeline_func(good_adjectives, bad_adjectives, uc_group, lc_group, embeddings_file)
        else:
            score = pipeline_func(good_adjectives, bad_adjectives, uc_group, lc_group)

        # Print the score with model name
        print(f"{model_name}: {score:.5f}")
    
    print("------------------------------------------------------------------------------------")


