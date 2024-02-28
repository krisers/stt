from utils import get_text_from_srt
INPUT=get_text_from_srt('b99s04e05.srt')

from utils import SummarizerQwen
sq = SummarizerQwen(qwen_model_name="Qwen/Qwen1.5-7B-Chat",device='cpu')

task = "summarize as a teaser synopsis in 3 sentences"
prompt_system = "You are a helpful assistant."

summary = sq.summarize(task=task,input_text=INPUT,prompt_sys=prompt_system)
print(summary)