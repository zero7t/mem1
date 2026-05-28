import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import string
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TAG_RE = re.compile(r"</?([a-zA-Z_][\w-]*)>")


def normalize_api_base(api_base: str) -> str:
    api_base = api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"
    return api_base


def auth_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_json(url: str, payload: Dict[str, Any], api_key: Optional[str], timeout: int, retries: int) -> Dict[str, Any]:
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=auth_headers(api_key), json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(str(last_error))


def call_completion(
    api_base: str,
    api_key: Optional[str],
    model: str,
    prompt: str,
    stop: List[str],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    data = post_json(f"{api_base}/completions", payload, api_key, timeout, retries)
    choice = data["choices"][0]
    text = choice.get("text", "").strip()
    stop_reason = choice.get("stop_reason") or choice.get("stop")
    finish_reason = choice.get("finish_reason")
    if stop_reason in stop and not text.endswith(stop_reason):
        text += stop_reason
    elif finish_reason == "stop" and not any(text.endswith(item) for item in stop):
        for item in stop:
            open_tag = f"<{item[2:]}"
            if item.startswith("</") and open_tag in text:
                text += item
                break
    return text


def call_chat_completion(
    api_base: str,
    api_key: Optional[str],
    model: str,
    content: str,
    stop: List[str],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    data = post_json(f"{api_base}/chat/completions", payload, api_key, timeout, retries)
    choice = data["choices"][0]
    text = choice.get("message", {}).get("content", "").strip()
    stop_reason = choice.get("stop_reason") or choice.get("stop")
    finish_reason = choice.get("finish_reason")
    if stop_reason in stop and not text.endswith(stop_reason):
        text += stop_reason
    elif finish_reason == "stop" and not any(text.endswith(item) for item in stop):
        for item in stop:
            open_tag = f"<{item[2:]}"
            if item.startswith("</") and open_tag in text:
                text += item
                break
    return text


def retrieve(search_url: str, query: str, topk: int, timeout: int, retries: int) -> Tuple[str, List[Dict[str, Any]]]:
    payload = {"queries": [query], "topk": topk, "return_scores": True}
    data = post_json(search_url, payload, None, timeout, retries)
    result = data.get("result", [])
    if not result or not isinstance(result[0], list):
        raise RuntimeError(f"bad retrieval response: {str(data)[:1000]}")
    docs = result[0]
    lines = []
    for idx, item in enumerate(docs):
        doc = item.get("document", item)
        contents = doc.get("contents", "")
        title = contents.split("\n")[0]
        text = "\n".join(contents.split("\n")[1:])
        lines.append(f"Doc {idx + 1}(Title: {title}) {text}")
    return "\n".join(lines) + "\n", docs


def scalar_to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [scalar_to_python(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: scalar_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scalar_to_python(v) for v in value]
    return value


def extract_prompt(row: pd.Series) -> str:
    prompt = scalar_to_python(row["prompt"])
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        if isinstance(first, dict) and "content" in first:
            return str(first["content"])
    if isinstance(prompt, dict) and "content" in prompt:
        return str(prompt["content"])
    return str(prompt)


def flatten_answers(value: Any) -> List[str]:
    value = scalar_to_python(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        if "target" in value:
            return flatten_answers(value["target"])
        return [json.dumps(value, ensure_ascii=False)]
    if isinstance(value, list):
        if not value:
            return []
        if all(not isinstance(v, (list, dict)) for v in value):
            return [str(v) for v in value]
        flattened = []
        for item in value:
            item_answers = flatten_answers(item)
            if len(item_answers) == 1:
                flattened.append(item_answers[0])
            elif item_answers:
                flattened.append("|".join(item_answers))
        return flattened
    return [str(value)]


def extract_ground_truth(row: pd.Series) -> List[List[str]]:
    row_dict = scalar_to_python(row.to_dict())
    reward_model = row_dict.get("reward_model") or {}
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, dict) and "target" in ground_truth:
            raw = scalar_to_python(ground_truth["target"])
            if isinstance(raw, list):
                targets = []
                for item in raw:
                    answers = flatten_answers(item)
                    targets.append(answers or [str(item)])
                return targets
            return [flatten_answers(raw)]
    for key in ["golden_answers", "answer", "answers"]:
        if key in row_dict:
            answers = flatten_answers(row_dict[key])
            return [[answer] for answer in answers]
    return [[]]


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_matches(prediction: str, golds: List[str]) -> bool:
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False
    for gold in golds:
        gold_norm = normalize_answer(gold)
        if not gold_norm:
            continue
        if pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm:
            return True
    return False


def score_answer(answer: Optional[str], targets: List[List[str]]) -> Tuple[int, List[Dict[str, Any]]]:
    if answer is None:
        return 0, []
    predictions = [part.strip() for part in str(answer).split(";")]
    details = []
    exact_match = 0
    for idx, golds in enumerate(targets):
        pred = predictions[idx] if idx < len(predictions) else ""
        ok = answer_matches(pred, golds)
        exact_match += int(ok)
        details.append({"prediction": pred, "gold": golds, "match": ok})
    return exact_match, details


def extract_one(pattern: re.Pattern, text: str) -> Optional[str]:
    matches = pattern.findall(text)
    if len(matches) != 1:
        return None
    return matches[0].strip()


def validate_response(text: str, turn_idx: int, max_turns: int, previous_queries: set, max_query_chars: int) -> Dict[str, Any]:
    errors = []
    think_matches = THINK_RE.findall(text)
    search_matches = SEARCH_RE.findall(text)
    answer_matches_ = ANSWER_RE.findall(text)
    tags = TAG_RE.findall(text)
    allowed_tags = {"think", "search", "answer"}

    unknown_tags = sorted({tag for tag in tags if tag not in allowed_tags})
    if unknown_tags:
        errors.append(f"unknown_assistant_tags:{','.join(unknown_tags)}")
    if len(think_matches) != 1 or not think_matches[0].strip():
        errors.append("missing_or_multiple_think")
    if len(search_matches) + len(answer_matches_) != 1:
        errors.append("must_have_exactly_one_search_or_answer")
    if len(search_matches) > 1:
        errors.append("multiple_search_tags")
    if len(answer_matches_) > 1:
        errors.append("multiple_answer_tags")

    action = None
    content = None
    if len(search_matches) == 1 and not answer_matches_:
        action = "search"
        content = search_matches[0].strip()
        norm_query = normalize_answer(content)
        if not content:
            errors.append("empty_search_query")
        if len(content) > max_query_chars:
            errors.append("search_query_too_long")
        if "\n" in content:
            errors.append("search_query_contains_newline")
        if TAG_RE.search(content):
            errors.append("search_query_contains_tag")
        if norm_query in previous_queries:
            errors.append("repeated_search_query")
        if turn_idx == max_turns - 1:
            errors.append("search_on_last_turn")
    elif len(answer_matches_) == 1 and not search_matches:
        action = "answer"
        content = answer_matches_[0].strip()
        if not content:
            errors.append("empty_answer")

    action_text = None
    if action == "search":
        action_text = f"<search>{content}</search>"
    elif action == "answer":
        action_text = f"<answer>{content}</answer>"

    return {
        "valid": not errors,
        "errors": errors,
        "action": action,
        "content": content,
        "action_text": action_text,
        "think": f"<think>{think_matches[0].strip()}</think>" if len(think_matches) == 1 else "",
    }


def build_mem1_prompt(tokenizer: Any, initial_prompt: str, content: str) -> str:
    messages = [{"role": "user", "content": initial_prompt}, {"role": "assistant", "content": content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    suffix = "<|im_end|>\n"
    if prompt.endswith(suffix):
        prompt = prompt[:-len(suffix)]
    return prompt


def make_sft_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples = []
    q = record["q"].strip()
    previous_step = ""
    step = 0
    while f"t{step}" in record and f"r{step}" in record:
        user_content = q if step == 0 else q + previous_step
        assistant = (record[f"t{step}"].strip() + record[f"r{step}"].strip()).strip()
        examples.append({"messages": [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant}]})
        if f"i{step}" in record and record[f"i{step}"]:
            previous_step = record[f"t{step}"].strip() + record[f"r{step}"].strip() + record[f"i{step}"].strip()
        else:
            previous_step = record[f"t{step}"].strip() + record[f"r{step}"].strip()
        step += 1
    return examples


def load_done_hashes(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if "hash" in item:
                    done.add(item["hash"])
            except Exception:
                continue
    return done


def generate_one(row_tuple: Tuple[int, pd.Series], args: argparse.Namespace, tokenizer: Any = None) -> Dict[str, Any]:
    index, row = row_tuple
    prompt = extract_prompt(row)
    targets = extract_ground_truth(row)
    row_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    api_base = normalize_api_base(args.api_base)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    record: Dict[str, Any] = {
        "index": int(index) if isinstance(index, (int, np.integer)) else str(index),
        "hash": row_hash,
        "q": prompt,
        "Golden_answer": targets,
        "Exact_match": 0,
        "answer": None,
        "process_valid": False,
        "accepted": False,
        "errors": [],
        "search_queries": [],
    }

    cur_obs = "" if args.inference_style == "mem1" else prompt
    previous_queries = set()

    for turn_idx in range(args.max_turns):
        stop = ["</answer>"] if turn_idx == args.max_turns - 1 else ["</search>", "</answer>"]
        try:
            if args.inference_style == "mem1":
                completion_prompt = build_mem1_prompt(tokenizer, prompt, cur_obs)
                response = call_completion(
                    api_base, api_key, args.model, completion_prompt, stop,
                    args.temperature, args.max_tokens, args.timeout, args.retries,
                )
            else:
                response = call_chat_completion(
                    api_base, api_key, args.model, cur_obs, stop,
                    args.temperature, args.max_tokens, args.timeout, args.retries,
                )
        except Exception as exc:
            record["errors"].append(f"generation_error:{exc}")
            break

        validation = validate_response(response, turn_idx, args.max_turns, previous_queries, args.max_query_chars)
        record[f"t{turn_idx}"] = validation["think"]
        record[f"r{turn_idx}"] = validation["action_text"] or response
        record[f"raw_response{turn_idx}"] = response

        if not validation["valid"]:
            record["errors"].extend([f"turn_{turn_idx}:{err}" for err in validation["errors"]])
            break

        assistant_step = record[f"t{turn_idx}"] + record[f"r{turn_idx}"]
        if validation["action"] == "answer":
            record["answer"] = validation["content"]
            break

        query = validation["content"] or ""
        previous_queries.add(normalize_answer(query))
        record["search_queries"].append(query)

        try:
            search_results, docs = retrieve(args.search_url, query, args.topk, args.timeout, args.retries)
        except Exception as exc:
            record["errors"].append(f"turn_{turn_idx}:retrieval_error:{exc}")
            break

        if len(docs) < args.topk:
            record["errors"].append(f"turn_{turn_idx}:retrieval_returned_{len(docs)}_docs")
            break

        turns_left = args.max_turns - turn_idx - 1
        if turns_left > 1:
            hint = f"[HINT]You have {turns_left} turns left.[/HINT]"
        elif turns_left == 1:
            hint = "[HINT]You have 1 turn left. You must answer the question in the next turn.[/HINT]"
        else:
            hint = ""
        information = f"\n\n<information>{hint}\n\n{search_results.strip()}</information>\n\n"
        record[f"i{turn_idx}"] = information

        if args.inference_style == "mem1":
            cur_obs = assistant_step + information
        else:
            cur_obs = prompt + assistant_step + information

    exact_match, answer_details = score_answer(record.get("answer"), targets)
    record["Exact_match"] = exact_match
    record["answer_details"] = answer_details
    required_matches = len(targets) if args.min_exact_match < 0 else args.min_exact_match
    if args.require_search and not record["search_queries"]:
        record["errors"].append("no_search_tool_call")
    record["process_valid"] = len(record["errors"]) == 0 and record.get("answer") is not None
    record["accepted"] = record["process_valid"] and exact_match >= required_matches
    return record


def write_sft_json(raw_path: Path, sft_path: Path) -> int:
    examples = []
    if raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("accepted"):
                    examples.extend(make_sft_messages(item))
    with sft_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-quality MEM1 SFT trajectories with API generation and tool/result checks.")
    parser.add_argument("--data_file", default="/root/paddlejob/workspace/mem1/MEM1/Mem1/train/data/nq_hotpotqa_train_multi_2/train.parquet")
    parser.add_argument("--output_jsonl", default="/root/paddlejob/workspace/mem1/MEM1/Mem1/inference/distill/outputs/sft_trajectories_raw.jsonl")
    parser.add_argument("--sft_output", default="/root/paddlejob/workspace/mem1/MEM1/Mem1/inference/distill/outputs/sft_train.json")
    parser.add_argument("--api_base", default="http://127.0.0.1:8014/v1")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--model", default="/root/paddlejob/workspace/mem1/MEM1/assets/models/Mem-Lab__Qwen2.5-7B-RL-RAG-Q2-EM-Release")
    parser.add_argument("--tokenizer_path", default="/root/paddlejob/workspace/mem1/MEM1/assets/models/Qwen__Qwen2.5-7B")
    parser.add_argument("--search_url", default="http://127.0.0.1:8013/retrieve")
    parser.add_argument("--inference_style", choices=["mem1", "normal"], default="mem1")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_turns", type=int, default=6)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max_query_chars", type=int, default=256)
    parser.add_argument("--min_exact_match", type=int, default=-1, help="-1 means all ground-truth answers must match.")
    parser.add_argument("--require_search", action="store_true", default=True, help="Require at least one valid search tool call before accepting a trajectory.")
    parser.add_argument("--allow_no_search", dest="require_search", action="store_false", help="Allow direct-answer trajectories without search.")
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    raw_path = Path(args.output_jsonl)
    sft_path = Path(args.sft_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if args.inference_style == "mem1":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, trust_remote_code=True)

    df = pd.read_parquet(args.data_file)
    if args.offset:
        df = df.iloc[args.offset:]
    if args.limit is not None:
        if args.limit <= 0:
            df = df.iloc[:0]
        else:
            df = df.iloc[:args.limit]

    done_hashes = set() if args.no_resume else load_done_hashes(raw_path)
    rows = []
    for index, row in df.iterrows():
        prompt = extract_prompt(row)
        row_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if row_hash not in done_hashes:
            rows.append((index, row))

    lock = threading.Lock()
    accepted = 0
    total = 0

    def run_and_write(row_tuple: Tuple[int, pd.Series]) -> Dict[str, Any]:
        item = generate_one(row_tuple, args, tokenizer)
        with lock:
            with raw_path.open("a", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        return item

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_and_write, row_tuple) for row_tuple in rows]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="generating"):
            item = future.result()
            total += 1
            accepted += int(bool(item.get("accepted")))

    sft_count = write_sft_json(raw_path, sft_path)
    print(json.dumps({
        "processed_this_run": total,
        "accepted_this_run": accepted,
        "raw_output": str(raw_path),
        "sft_output": str(sft_path),
        "sft_examples": sft_count,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
