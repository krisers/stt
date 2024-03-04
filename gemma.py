import time
import argparse
from utils import get_text_from_srt
from utils import SummarizerQwen
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKEN = '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'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', metavar='str', nargs=1, default=['Qwen/Qwen1.5-7B-Chat'], help='Model name')
    parser.add_argument('--text_filename', metavar='str', nargs=1, default=['b99s04e05.srt'], help='Transcript file to be summarized')
    parser.add_argument('--device', metavar='str', nargs=1, default=['cuda'], help='cpu or cuda')
    parser.add_argument('--prompt_ins', metavar='str', nargs=1, default=['You are a helpful assistant.'], help='Instruction for prompt behaviour')
    parser.add_argument('--task', metavar='str', nargs=1, default=['"What do you know about Edgar Allan Poe?"'], help='task for LLM')

    args = parser.parse_args()

    model_name = args.model_name[0]
    print(f'Model_name: \t{model_name}')
    text_fn = args.text_filename[0]
    print(f'text_filename: \t{text_fn}')
    device = args.device[0]
    print(f'Device: \t{device}')
    prompt_ins = args.prompt_ins[0]
    print(f'Prompt_ ins: \t{prompt_ins}')
    task = args.task[0]
    print(f'task: \t{task}')


    INPUT=get_text_from_srt(text_fn)
    out_fn= f'{text_fn[:-4]}_summary_{model_name[-7:-5]}_{device}.txt'
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it",token=TOKEN)
    model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it",token=TOKEN)

    #input_text = "What do you know about Edgar Allan Poe?"

    input_ids = tokenizer(f'Summerize in one sentence: { INPUT}', return_tensors="pt")

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


    #task = "summarize as a teaser synopsis in 3 sentences"

