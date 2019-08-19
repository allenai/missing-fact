import re
from typing import List, Any, Dict

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

stop_words_set = set(stopwords.words('english'))

EN_PREFIX = "/c/en/"
EN_CONCEPT_RE = re.compile("/c/en/([^/]*).*")


def conceptnet_entity(ent):
    m = EN_CONCEPT_RE.match(ent)
    if m:
        return m.group(1).lower()
    else:
        return None


def tokenize_str(input_str):
    return [stemmer.stem(str) for str in re.split("[\W_]+", input_str.lower())
            if str not in stop_words_set]


def tokenize_and_stem_str(input_str):
    return [(str, stemmer.stem(str)) for str in re.split("[\W_]+", input_str.lower())
            if str not in stop_words_set]


def accept_relation(rel, ignore_related=False):
    if rel == "/r/Antonym":
        return False
    if rel.startswith("/r/dbpedia/"):
        return False
    if rel.startswith("/r/Etymologically"):
        return False
    if ignore_related and rel.startswith("/r/RelatedTo"):
        return False
    return True


def convert_relation_to_string(rel):
    if rel.lower() == "/r/isa":
        return "a type of"
    if rel.lower() == "/r/none":
        return "not related to"
    return " ".join([x.lower() for x in split_relation(rel)])

def convert_entity_to_string(ent):
    # clean entity name if needed
    ent_name = conceptnet_entity(ent)
    if ent_name is None:
        ent_name = ent
    return " ".join(ent_name.split("_"))

def split_relation(rel):
    if rel.startswith("/r/"):
        rel = rel[len("/r/"):]
    return re.split("[\W_]+", re.sub('([a-z])([A-Z])', r'\1 \2', rel))


def load_kbtuples_map(conceptnet_kb_path, ignore_related=False) -> (List[Any], Dict[str, List[int]]):
    conceptnet_triples = []
    conceptnet_map = {}
    with open(conceptnet_kb_path, 'r') as kb_file:
        for line in kb_file:
            field = line.strip().split("\t")
            ent1 = conceptnet_entity(field[1])
            rel = field[0]
            ent2 = conceptnet_entity(field[2])
            if ent1 and ent2 and accept_relation(rel, ignore_related):
                ent1_toks = tokenize_str(ent1)
                ent2_toks = tokenize_str(ent2)
                conceptnet_triples.append((ent1, rel, ent2))
                for ent1_tok in ent1_toks:
                    if ent1_tok not in conceptnet_map:
                        conceptnet_map[ent1_tok] = []
                    conceptnet_map[ent1_tok].append(len(conceptnet_triples) - 1)
                for ent2_tok in ent2_toks:
                    if ent2_tok not in conceptnet_map:
                        conceptnet_map[ent2_tok] = []
                    conceptnet_map[ent2_tok].append(len(conceptnet_triples) - 1)
    return conceptnet_triples, conceptnet_map


def retrieve_scored_tuples(ent1, ent2, kbtuples, kbmap, max=100):
    match_set = set()
    num_relations = 0
    ent1_toks = tokenize_and_stem_str(ent1)
    ent2_toks = tokenize_and_stem_str(ent2)
    additional_tuples = []
    for (ent1_orig, ent1_tok) in ent1_toks:
        ent1_set = set(kbmap.get(ent1_tok, []))
        for (ent2_orig, ent2_tok) in ent2_toks:
            match_found = False
            if ent2_tok == ent1_tok:
                additional_tuples.append((ent1_orig, "/r/SameAs", ent2_orig))
                num_relations += 1
                continue
            ent2_set = set(kbmap.get(ent2_tok, []))
            ent1_ent2_inter = ent1_set.intersection(ent2_set)
            for idx in ent1_ent2_inter:
                kb_ent1_toks = tokenize_str(kbtuples[idx][0])
                kb_ent2_toks = tokenize_str(kbtuples[idx][2])
                if ent1_tok in kb_ent1_toks and ent2_tok in kb_ent2_toks:
                    match_set.add(idx)
                    match_found = True
                elif ent1_tok in kb_ent2_toks and ent2_tok in kb_ent1_toks:
                    match_set.add(idx)
                    match_found = True
            if match_found:
                num_relations += 1
    scored_tuples = []
    for tuple in additional_tuples:
        scored_tuples.append((tuple, 1.0))
    if not len(scored_tuples) and not len(match_set):
        return [((ent1, "/r/NONE", ent2), 0.0)]
    for tupleidx in match_set:
        ent_toks = set([x[1] for x in ent1_toks] + [x[1] for x in ent2_toks])
        kb_toks = set(tokenize_str(kbtuples[tupleidx][0]) + tokenize_str(kbtuples[tupleidx][2]))
        if len(ent_toks) or len(kb_toks):
            score = len(ent_toks.intersection(kb_toks)) / len(ent_toks.union(kb_toks))
            scored_tuples.append((kbtuples[tupleidx], score))
    scored_tuples.sort(key=lambda x: -x[1])
    return scored_tuples[:max]
