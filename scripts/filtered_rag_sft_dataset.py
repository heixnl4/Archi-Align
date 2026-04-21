import json

input_path = "./rag_sft_dataset.jsonl"
output_path = "./filtered_rag_sft_dataset.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line.strip())
        filtered_data = {
            "instruction": data["instruction"],
            "output": data["output"]
        }
        outfile.write(json.dumps(filtered_data, ensure_ascii=False) + "\n")