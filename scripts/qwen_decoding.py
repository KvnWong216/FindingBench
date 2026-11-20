"""Qwen3-VL Instruct official evaluation sampling settings.

Source: https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction
Presence penalty matches vLLM: subtract once for tokens in the generated
continuation, regardless of frequency; do not penalize prompt/history tokens.
"""
from transformers import LogitsProcessor, LogitsProcessorList

SETTINGS = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                repetition_penalty=1.0, presence_penalty=1.5,
                seed=3407, max_new_tokens=32768)


class GeneratedPresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length, penalty=1.5):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        scores = scores.clone()
        for row in range(input_ids.shape[0]):
            seen = input_ids[row, self.prompt_length:].unique()
            scores[row, seen] -= self.penalty
        return scores


def generation_kwargs(prompt_length):
    return dict(do_sample=True, temperature=SETTINGS['temperature'],
                top_p=SETTINGS['top_p'], top_k=SETTINGS['top_k'],
                repetition_penalty=SETTINGS['repetition_penalty'], num_beams=1,
                logits_processor=LogitsProcessorList([
                    GeneratedPresencePenalty(prompt_length, SETTINGS['presence_penalty'])]))
