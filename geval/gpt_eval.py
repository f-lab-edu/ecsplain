import argparse
import json
import os
import time

import openai
import tqdm

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--prompt_fp", type=str, default="prompts/summeval/con_detailed.txt")
    argparser.add_argument("--save_fp", type=str, default="results/gpt4_con_detailed_openai.json")
    argparser.add_argument("--summeval_fp", type=str, default="data/summeval.json")
    argparser.add_argument("--key", type=str, default=None)
    argparser.add_argument("--model", type=str, default="gpt-4-0613")
    args = argparser.parse_args()
    if args.key is None:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    with open(args.summeval_fp) as read_fp:
        summeval = dict()
        if "jsonl" in args.summeval_fp:
            for line in read_fp.readlines():
                js = json.loads(line)
                summeval[js["id"]] = js
        else:
            summeval = json.load(read_fp)
    prompt = open(args.prompt_fp).read()

    ct, ignore = 0, 0

    new_json = []
    for instance in tqdm.tqdm(summeval):
        source = instance["source"]
        explanation = instance["explanation"].split("[4단계]")[-1]
        instance["final_explanation"] = explanation
        cur_prompt = prompt.replace("{{Document}}", source).replace("{{Explanation}}", explanation)
        instance["prompt"] = cur_prompt
        while True:
            try:
                _response = client.responses.create(
                    model=args.model, input=cur_prompt, reasoning={"effort": "high"}
                )
                time.sleep(0.5)

                all_responses = _response.output_text
                instance["evaluation"] = all_responses
                new_json.append(instance)
                ct += 1
                break
            except Exception as e:
                print(e)
                if "limit" in str(e):
                    time.sleep(2)
                else:
                    ignore += 1
                    print("ignored", ignore)

                    break

    base_dir = "/".join(args.save_fp.split("/")[:-1])
    os.makedirs(base_dir, exist_ok=True)
    print("ignored total", ignore)
    with open(args.save_fp, "w", encoding="utf-8") as write_fp:
        json.dump(new_json, write_fp, ensure_ascii=False, indent=4)
