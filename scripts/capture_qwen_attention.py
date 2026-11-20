"""Read-only per-token single-head attention instrumentation for Qwen3-VL.

SDPA still computes and returns the model output unchanged. We independently
evaluate the last query's softmax from the live, post-RoPE Q/K and actual mask.
Each generated token retains one actual head: the largest image-attention mass
across all language layers. No head, layer, or token averaging.
This is attention, not a causal attribution or a gaze measurement.
"""
from pathlib import Path
import json
import uuid


def last_query_weights(module, query, key, mask, scaling):
    import torch
    # GQA broadcasting avoids materializing the repeated full-length key tensor.
    b, heads, _, dim = query.shape
    grouped = query[:, :, -1:, :].float().reshape(
        b, key.shape[1], module.num_key_value_groups, 1, dim)
    scores = (grouped @ key.float().unsqueeze(2).transpose(-2, -1))
    scores = scores.reshape(b, heads, 1, key.shape[-2]) * scaling
    if mask is not None:
        if mask.ndim != 4:
            raise ValueError("Only explicit 4-D or absent SDPA masks are supported")
        last_mask = mask[:, :, -1:, :key.shape[-2]]
        if last_mask.dtype == torch.bool:
            scores = scores.masked_fill(~last_mask, float("-inf"))
        else:
            scores = scores + last_mask.float()
    # Unpadded single-sequence prefill / cached decode: the last query can
    # attend to all existing keys. Future keys are not present in this cache.
    return torch.softmax(scores, dim=-1)[0, :, 0, :]


class QwenAttentionRecorder:
    def __init__(self, model, processor, output):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        self.model, self.processor, self.output = model, processor, Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        layers = [m for m in model.modules() if type(m).__name__ == "Qwen3VLTextAttention"]
        if not layers:
            raise ValueError("Attention recorder supports Qwen3-VL text attention only")
        self.layers = layers
        self.layer = layers[-1]
        if self.layer.config._attn_implementation != "sdpa":
            raise ValueError("Recorder requires the existing SDPA backend")
        self.original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.active = False
        def observed(module, query, key, value, attention_mask, **kw):
            result = self.original(module, query, key, value, attention_mask, **kw)
            if self.active and module in self.layers:
                weights = last_query_weights(module, query, key, attention_mask,
                                             kw.get("scaling", module.scaling))
                image_weights = weights[:, self.positions]
                masses = image_weights.sum(dim=-1)
                head = int(masses.argmax().item())
                mass = float(masses[head].item())
                if module is self.layers[0]:
                    self.best = None
                if self.best is None or mass > self.best[0]:
                    self.best = (mass, int(module.layer_idx), head,
                                 image_weights[head].detach().cpu().numpy(),
                                 float(weights[head].sum().item()))
                if module is self.layers[-1]:
                    mass, layer, head, row, norm = self.best
                    self.rows.append(row)
                    self.selected_heads.append((layer, head, mass))
                    self.normalization.append(norm)
            return result
        ALL_ATTENTION_FUNCTIONS.register("sdpa", observed)

    def begin(self, inputs):
        import torch
        ids = inputs["input_ids"]
        if ids.shape[0] != 1 or not bool(inputs["attention_mask"].all()):
            raise ValueError("Recorder requires one unpadded sequence")
        self.positions = torch.where(ids[0] == self.model.config.image_token_id)[0]
        self.grids = inputs["image_grid_thw"].detach().cpu().tolist()
        self.merge = self.model.config.vision_config.spatial_merge_size
        self.rows, self.normalization, self.selected_heads = [], [], []
        expected = sum(t*h*w//self.merge**2 for t,h,w in self.grids)
        if expected != len(self.positions):
            raise ValueError("Image token grid does not match image token positions")
        self.active = True

    def finish(self, generated):
        import numpy as np
        self.active = False
        if len(self.rows) != len(generated):
            raise RuntimeError("Attention/generation token alignment mismatch")
        weights = np.stack(self.rows)
        if not np.isfinite(weights).all() or not np.allclose(self.normalization, 1, atol=1e-5):
            raise RuntimeError("Attention normalization failed")
        stem = self.output / uuid.uuid4().hex
        token_ids = generated.detach().cpu().numpy()
        np.savez_compressed(stem.with_suffix(".npz"), image_attention=weights, generated_token_ids=token_ids,
                            image_positions=self.positions.cpu().numpy(), image_grid_thw=np.asarray(self.grids),
                            spatial_merge_size=np.asarray(self.merge))
        # Select output-value tokens only, never all reasoning or JSON punctuation.
        import re
        text = self.processor.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        ends = [len(self.processor.tokenizer.decode(token_ids[:i+1].tolist(),
                skip_special_tokens=True)) for i in range(len(token_ids))]
        starts = [0] + ends[:-1]
        action_start = text.rfind('"action"')
        selections = []
        if action_start >= 0:
            pattern = r'"(skill|distance_cm|angle_deg|yaw|pitch|x|y)"\s*:\s*"?([^",}\s]+)'
            for match in re.finditer(pattern, text[action_start:]):
                lo, hi = match.span(2); lo += action_start; hi += action_start
                for i, (start, end) in enumerate(zip(starts, ends)):
                    if start < hi and end > lo:
                        layer, head, mass = self.selected_heads[i]
                        selections.append(dict(field=match.group(1), value=match.group(2),
                            token_index=i, token_text=text[start:end], layer=layer, head=head,
                            image_attention_mass=mass))
        meta = dict(method="single head per output token; live post-RoPE QK softmax",
                    selection="maximum image attention mass across every language layer/head; ties use first index",
                    aggregation="none", selected_queries=selections,
                    selected_heads=self.selected_heads,
                    note="Predictive query for the named output token (one-token shift). Exploratory head selection, not validated localization or causal attribution.",
                    sources=["https://github.com/meng-du/VL-InterpreT/",
                             "https://github.com/seilk/LocalizationHeads"],
                    npz=str(stem.with_suffix(".npz")), grids=self.grids,
                    spatial_merge_size=self.merge, generated_tokens=len(token_ids),
                    generated_token_strings=self.processor.tokenizer.convert_ids_to_tokens(token_ids.tolist()),
                    all_key_probability_sum_min=min(self.normalization),
                    all_key_probability_sum_max=max(self.normalization))
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return meta
