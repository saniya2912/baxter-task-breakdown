import json
from PIL import Image
import torch
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration

MODEL_ID = "microsoft/kosmos-2-patch14-224"
IMAGE_PATH = "clothes_example.jpeg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def best_effort_json_from_caption(caption: str) -> dict:
    """
    Minimal heuristic: if caption mentions 'white' or 'light' treat as whites, else colours.
    Later we'll replace this with a second VLM prompt asking it to output strict JSON.
    """
    cap = caption.lower()
    items = []

    # Very simple item extraction heuristic (you'll improve later)
    # If caption doesn't list multiple items, we return one "unknown garment".
    item_name = "garment"
    dominant_color = "unknown"

    # crude color guesses
    for c in ["white", "cream", "beige", "grey", "gray", "black", "blue", "red", "green", "yellow", "pink", "purple", "brown", "orange"]:
        if c in cap:
            dominant_color = c
            break

    if any(w in cap for w in ["white", "cream", "light gray", "light grey"]):
        bin_name = "whites"
        conf = 0.7
    elif dominant_color != "unknown":
        bin_name = "colours"
        conf = 0.6
    else:
        bin_name = "colours"
        conf = 0.4

    items.append({
        "item": item_name,
        "dominant_color": dominant_color,
        "bin": bin_name,
        "confidence": float(conf)
    })

    return {"items": items, "notes": f"Caption-based decision. Caption: {caption}"}


def main():
    print(f"[info] device={DEVICE}, dtype={DTYPE}, model={MODEL_ID}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Kosmos2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    image = Image.open(IMAGE_PATH).convert("RGB")

    # Kosmos-2 uses a special prompt style; simplest is descriptive captioning
    prompt = "<image> Describe the clothing items and their colors."

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Convert caption -> a first-pass laundry decision (keeps progress moving)
    result = best_effort_json_from_caption(caption)

    print("\n--- CAPTION ---\n", caption)
    print("\n--- JSON OUTPUT ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
