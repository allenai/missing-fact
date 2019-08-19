import re

## pre-compiled regular expression for decomposing a multiple-choice question
question_regex = re.compile('(.*)\(A\)(.*)\(B\)(.*)\(C\)(.*)\(D\)(.*)')


## decompose a multiple choice question into its stem and (randomly shuffled) answer choices

def decompose_question(raw_question):
    try:
        # apply regex to extract question stem and answer choices
        question_parts = question_regex.search(raw_question).groups()
        # remove white space
        result = [s.strip() for s in question_parts]
        # put together question stem and choices
        question_stem = result[0]
        choice_strings = result[1:5]
        return question_stem, choice_strings

    except Exception as e:
        print('ERROR decomposing the following question:\n  {}'.format(raw_question))
        print('EXCEPTION: {}'.format(e))
        return None


def create_arc_json(qid, question, choices, predictions, fact=None, debug_info=None):
    prediction_json = []
    choices_json = []
    selected_answer = ""
    selected_idx = -1
    top_prob = 0
    for idx, choice in enumerate(choices):
        if predictions[idx] > top_prob:
            top_prob = predictions[idx]
            selected_answer = choice
            selected_idx = idx
        prediction_json.append({
            "label": chr(ord('A') + idx),
            "text": choice,
            "score": float("{0:.6f}".format(predictions[idx]))
        })
        choices_json.append({
            "label": chr(ord('A') + idx),
            "text": choice
        })
    output_json = {
        "id": qid,
        "question": {
            "stem": question,
            "choices": choices_json,
        },
        "fact": fact,
        "prediction": {
            "choices": prediction_json
        },
        "selected_answer": selected_answer,
        "selected_idx": selected_idx,
        "debug_info": debug_info
    }
    return output_json
