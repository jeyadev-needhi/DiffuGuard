import argparse
import logging

from utils import Refiner


def main():
    parser = argparse.ArgumentParser(
        description="Refine WildJailbreak prompts using Qwen2.5-7B-Instruct"
    )

    parser.add_argument(
        "--hf_model_path",
        required=True,
        help="Hugging Face model name or local model path"
    )

    parser.add_argument(
        "--prompt_template_path",
        required=True
    )

    parser.add_argument(
        "--attack_prompt",
        required=True,
        help="Input JSON containing id, label, vanilla prompt, refined prompt"
    )

    parser.add_argument(
        "--output_json",
        required=True
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    refiner = Refiner(
        hf_model_path=args.hf_model_path,
        api_model_name=None,
        prompt_template_path=args.prompt_template_path,
        attack_prompt=args.attack_prompt,
        output_json=args.output_json,
        base_url=None,
        api_key=None,
    )

    refiner.run_refinement_hf(
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()
