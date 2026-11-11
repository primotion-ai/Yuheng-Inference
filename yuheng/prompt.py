from qwen_vl_utils import process_vision_info
import torch


def conversation(instruction, references, progress, query):
    content = [{"type": "text", "text": (
        f"Task: {instruction}\nThe following ordered reference keyframes define relative progress for this query. "
        "The first reference means 0% and the last means 100% of this reference segment, "
        "not necessarily completion of the entire task.\nBEGIN REFERENCE KEYFRAMES\n"
    )}]
    for i, (image, value) in enumerate(zip(references, progress), start=1):
        content.extend([
            {"type": "text", "text": f"Reference {i} of {len(references)}, progress {100 * value:.4f}%:\n"},
            {"type": "image", "image": image},
        ])
    content.extend([
        {"type": "text", "text": "\nEND REFERENCE KEYFRAMES\nCURRENT QUERY IMAGE (not a reference):\n"},
        {"type": "image", "image": query},
        {"type": "text", "text": "\nLocate the current state relative to the reference keyframes. Relative progress:"},
    ])
    return [{"role": "user", "content": content}]


def prepare_batch(processor, instruction, references, progress, queries):
    messages = [conversation(instruction, references, progress, image) for image in queries]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in messages]
    images, _ = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)
    inputs = processor(text=texts, images=images, padding=True, truncation=False,
                       do_resize=False, return_tensors="pt")
    end_token = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    positions = []
    for tokens in inputs["input_ids"]:
        ends = torch.where(tokens == end_token)[0]
        if ends.numel() != len(references) + 1:
            raise ValueError("Expected one image boundary per reference plus one query boundary")
        positions.append(ends[:-1])
    inputs["anchor_positions"] = torch.stack(positions)
    inputs["query_positions"] = inputs["attention_mask"].sum(-1) - 1
    inputs["anchor_progress"] = torch.tensor(progress, dtype=torch.float32).expand(len(queries), -1)
    return inputs
