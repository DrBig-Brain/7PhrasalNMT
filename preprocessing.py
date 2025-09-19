
# For English-Hindi translation, we use spaCy for English phrase extraction.
# If you want to do phrase extraction for Hindi, you may need to use a Hindi spaCy model or another NLP tool.
import spacy
nlp = spacy.load("en_core_web_sm")  # For English

PHRASE_TYPES = ['O', 'NP', 'VP', 'PP', 'ADJP', 'ADVP', 'CONJP', 'QP']

def extract_7_phrases(sentence):
    """
    Extract phrase tags for an English sentence using spaCy.
    For Hindi, you may need to implement a different function or use a Hindi NLP model.
    """
    doc = nlp(sentence)
    phrase_tags = []
    for token in doc:
        if token.dep_ in ['nsubj', 'pobj', 'dobj']:
            phrase_tags.append('NP')
        elif token.pos_ == 'VERB':
            phrase_tags.append('VP')
        elif token.dep_ == 'prep':
            phrase_tags.append('PP')
        elif token.pos_ == 'ADJ':
            phrase_tags.append('ADJP')
        elif token.pos_ == 'ADV':
            phrase_tags.append('ADVP')
        elif token.dep_ == 'cc':
            phrase_tags.append('CONJP')
        elif token.like_num:
            phrase_tags.append('QP')
        else:
            phrase_tags.append('O')
    return phrase_tags

def preprocess_with_phrases(x, y, minlength, maxlength):
    """
    Preprocess parallel English-Hindi data.
    x: list of English sentences
    y: list of Hindi sentences
    Returns: processed_x, processed_y, x_phrases (phrase tags for English)
    """
    processed_x, processed_y, x_phrases = [], [], []
    for xx, yy in zip(x, y):
        if minlength <= len(xx) <= maxlength and minlength <= len(yy) <= maxlength:
            processed_x.append(xx)
            processed_y.append(yy)
            x_phrases.append(extract_7_phrases(xx))  # Only for English
    return processed_x, processed_y, x_phrases
