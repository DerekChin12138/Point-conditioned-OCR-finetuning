"""Thin vLLM client for OvisOCR2 crop OCR (real-data labeling)."""

from __future__ import annotations

from PIL import Image


# Short prompt for cropped blocks (not full-page dump)
CROP_PROMPT = (
    "Transcribe the readable content in this cropped document region as Markdown. "
    "Format formulas as LaTeX and tables as HTML <table> if present. "
    "Output only the content of this crop — do not invent surrounding context."
)


class OvisCropClient:
    def __init__(self, model_name_or_path: str = "ATH-MaaS/OvisOCR2"):
        from vllm import LLM, SamplingParams

        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            gdn_prefill_backend="triton",
        )
        self.prompt = self.model.get_tokenizer().apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": CROP_PROMPT},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        self.sampling_params = SamplingParams(max_tokens=4096, temperature=0.0)

    def parse_crop(self, image: Image.Image) -> str:
        outputs = self.model.generate(
            [
                {
                    "prompt": self.prompt,
                    "multi_modal_data": {"image": image.convert("RGB")},
                    "mm_processor_kwargs": {
                        "images_kwargs": {
                            "min_pixels": 448 * 448,
                            "max_pixels": 1280 * 1280,
                        }
                    },
                }
            ],
            self.sampling_params,
        )
        return outputs[0].outputs[0].text.strip()
