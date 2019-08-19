from typing import List

import numpy
from numpy import random

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from missingfact.predictors.arc_output_utils import create_arc_json, decompose_question


@Predictor.register('arc_fact_span_kb')
class ArcFactSpanKbPredictor(Predictor):
    """
    """
    def predict(self, question: str, fact: str, span: str, relation:str) -> JsonDict:
        """
        Make a span relation predictor

        Parameters
        ----------
        question : ``str``
            A question about the content in the supplied paragraph.  The question must be answerable by a
            span in the paragraph.
        fact : ``str``
            A paragraph of information relevant to the question.

        Returns
        -------
        A dictionary that represents the prediction made by the system.  The answer string will be under the
        "best_span_str" key.
        """
        return self.predict_json({"fact": fact, "question": question, "span": span, "relation": relation})

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        """
        Override this method to create a formatted JSON
        :param instance:
        :return:
        """
        outputs = self._model.forward_on_instance(instance)
        output_json = self.format_output(outputs)
        return sanitize(output_json)

    def format_output(self, outputs):
        debug_info = outputs["metadata"].get("debug_info", {})
        debug_info["tuples"] = outputs["metadata"].get("selected_tuples", [])
        output_json = create_arc_json(outputs["metadata"]["id"],
                                      outputs["metadata"]["question_text"],
                                      outputs["metadata"]["choice_text_list"],
                                      outputs["label_probs"],
                                      outputs["metadata"]["fact_text"],
                                      debug_info)
        return output_json
    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        output_jsons = []
        for output in outputs:
            output_jsons.append(self.format_output(output))
        return sanitize(output_jsons)


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"question": "...", "fact": "..."}``.
        """
        if isinstance(json_dict["question"], dict):
            question_stem = json_dict["question"]["stem"]
            choices = [x["text"] for x in json_dict["question"]["choices"]]
        else:
            question_text = json_dict["question"]
            question_stem, choices = decompose_question(question_text)
        fact = json_dict.get("fact") or json_dict.get("fact1")
        span = json_dict.get("span") or json_dict.get("answer_spans")[0]
        spans = [span]
        if "relation" in json_dict:
            relations = [json_dict["relation"]]
        else:
            relations = None
        if "offset" in json_dict:
            offset = json_dict["offset"]
        elif "answer_starts" in json_dict:
            offset = json_dict["answer_starts"][0]
        else:
            offset = fact.index(span)
        if offset == -1:
            raise ValueError("Span: {} not found in fact: {}".format(span, fact))
        offsets = [offset] #[(offset, offset + len(span))]
        if "id" in json_dict:
            qid = json_dict["id"]
        else:
            qid = random.randint(100)
        prefetched_sentences = json_dict.get("prefetched_sentences", None)
        prefetched_indices = json_dict.get("prefetched_indices", None)
        if prefetched_sentences is not None:
            return self._dataset_reader.text_to_instance(qid, question_stem, choices, fact,
                                                         spans, relations, answer_starts=offsets,
                                                         prefetched_sentences=prefetched_sentences,
                                                         prefetched_indices=prefetched_indices)
        else:
            return self._dataset_reader.text_to_instance(qid, question_stem, choices, fact,
                                                         spans, relations, answer_starts=offsets)
