"""Generate two valid SKI v0.5 messages and sample replies."""
from pathlib import Path
import json
from rsd import make_sd
from ski_nl_exchange import export_message
messages=[export_message(make_sd(text),source_llm="LLM-A",target_llm="LLM-B")
          for text in ("GPUは並列計算に適している。","猫は窓辺で眠っている。")]
Path("data").mkdir(exist_ok=True)
Path("data/ski_dataset_a_example.json").write_text(json.dumps(messages,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
Path("data/ski_replies_example.json").write_text(json.dumps(["回答B-1","回答B-2"],ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("Created example A and sample receiver-generated text inputs.")
