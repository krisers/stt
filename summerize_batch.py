import os 
DEV = 'cpu'
MODEL = '7'
if __name__ == "__main__":
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='b99s04e03.srt' " )
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='b99s04e04.srt' " )
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='b99s04e05.srt' " )
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='b99s04e06.srt' " )
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='nanny.srt' " )
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='summer.srt' " )
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='lawandorder.srt' " )
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='arrival.srt' " )
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='theniceguys.srt' " )
    os.system(f"python test_qwen.py --device='{DEV}' --model_name='Qwen/Qwen1.5-{MODEL}B-Chat' --text_filename='theotherguys.srt' " )
