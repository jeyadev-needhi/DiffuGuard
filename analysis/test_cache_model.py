import sys
import torch
import inspect

sys.path.insert(0, "/home/jeyadev/dlm-research/LLaDA-DiffuGuard")

from transformers import AutoTokenizer, AutoModel

PATH = "/home/jeyadev/dlm-research/LLaDA-DiffuGuard"

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    PATH,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

model.eval()

print("MODEL:", type(model).__name__)
print()
print("FORWARD SIGNATURE:")
print(inspect.signature(model.forward))

prompt = "Explain why diffusion language models can generate tokens in parallel."

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

inputs = {
    k: v.to(model.device)
    for k, v in inputs.items()
}

print()
print("INPUT:", inputs["input_ids"].shape)

# --------------------------------------------------
# Normal inference
# --------------------------------------------------

print()
print("=" * 70)
print("TEST 1: NORMAL FORWARD")
print("=" * 70)

with torch.no_grad():

    out = model(
        input_ids=inputs["input_ids"],
        use_cache=False,
        return_dict=True
    )

print("NORMAL FORWARD: SUCCESS")
print("logits:", out.logits.shape)

# --------------------------------------------------
# Cache inference
# --------------------------------------------------

print()
print("=" * 70)
print("TEST 2: KV CACHE")
print("=" * 70)

try:

    with torch.no_grad():

        out = model(
            input_ids=inputs["input_ids"],
            use_cache=True,
            return_dict=True
        )

    print("KV CACHE: SUCCESS")

    pkv = out.past_key_values

    print("cache type:", type(pkv))
    print("cache layers:", len(pkv))

except Exception as e:

    print("KV CACHE: FAILED")
    print(type(e).__name__)
    print(str(e))

# --------------------------------------------------
# replace_position support
# --------------------------------------------------

print()
print("=" * 70)
print("TEST 3: REPLACE POSITION")
print("=" * 70)

if "replace_position" in inspect.signature(model.forward).parameters:
    print("replace_position: SUPPORTED")
else:
    print("replace_position: NOT SUPPORTED")

print()
print("DONE")
