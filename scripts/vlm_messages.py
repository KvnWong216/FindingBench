"""Lossless text turns; attach current images only to the final user turn."""


def text_messages(data):
    messages = data.get('messages')
    if messages is None:
        messages = [{'role': 'user', 'content': data['prompt']}]
    if not isinstance(messages, list) or not messages or len(messages) % 2 != 1:
        raise ValueError('Expected alternating user/assistant turns ending in user')
    result = []
    for i, message in enumerate(messages):
        role = 'user' if i % 2 == 0 else 'assistant'
        if message.get('role') != role or not isinstance(message.get('content'), str):
            raise ValueError('Invalid conversation role or text content')
        result.append({'role': role, 'content': message['content']})
    return result


def hf_messages(messages, pictures, labels):
    result = [{'role': m['role'], 'content': [{'type': 'text', 'text': m['content']}]}
              for m in messages]
    current = []
    for picture, label in zip(pictures, labels):
        current.extend([{'type': 'text', 'text': label}, {'type': 'image', 'image': picture}])
    result[-1]['content'] = current + result[-1]['content']
    return result


def minicpm_messages(messages, pictures, labels):
    result = [{'role': m['role'], 'content': [m['content']]} for m in messages]
    result[-1]['content'] = [*pictures, '\n'.join(labels), messages[-1]['content']]
    return result


def phi_messages(messages, image_count, labels):
    turns = []
    for i, message in enumerate(messages):
        prefix = ''
        if i == len(messages) - 1:
            prefix = ''.join(f'<|image_{j+1}|>' for j in range(image_count)) + '\n'.join(labels) + '\n'
        turns.append(f"<|{message['role']}|>" + prefix + message['content'] + '<|end|>')
    return ''.join(turns) + '<|assistant|>'


def validate_context_budget(input_tokens, output_tokens, context_limit):
    """Keep all input; bound new generation only by the model's native capacity."""
    if context_limit is None:
        return output_tokens
    remaining = context_limit - input_tokens
    if remaining <= 0:
        raise ValueError(
            f"ContextWindowExceeded: {input_tokens} input tokens leave no generation space "
            f"within the loaded model context limit of {context_limit}. "
            "History was not truncated and no action was generated."
        )
    return min(output_tokens, remaining)


def context_generation_exhausted(generated, effective_budget, requested_budget, eos_token_id):
    eos = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    return (effective_budget < requested_budget and len(generated) >= effective_budget
            and (not len(generated) or int(generated[-1]) not in eos))
