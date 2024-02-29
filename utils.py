import re 
import os
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import shutil
from PIL import Image, ImageDraw, ImageFont
from subprocess import run, CalledProcessError
from datetime import timedelta
from srt import Subtitle, compose,parse
import moviepy.editor as mp
import json 
import fasttext
import nltk




from sys import maxsize
import whisper
from pydub import AudioSegment

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer




SAMPLE_RATE = 16000

def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary
 
    Parameters
    ----------
    file: str
        The audio file to open
 
    sr: int
        The sample rate to resample the audio if necessary
 
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
 
    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    print('Loading audio')

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def preprocess(raw_text):

    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)
    words = letters_only_text.lower().split()

    return words

def chunks_video(filename,clen:int=30):
    print(f'Chunk length: {clen}')
    vid = AudioSegment.from_file(filename,filename[-3:])
    if clen ==-1:
        chunk_length = len(vid)
        chunk_max=1
    else:
        chunk_length = clen * 1000 # in ms 
        chunk_max = (len(vid)//chunk_length) +1


    print(f'Length video: {len(vid)}')
    print(f'Chunks video: {chunk_max}')

    folder_path = 'temp_' + filename.split('/')[-1]
    os.mkdir(folder_path)
    #one_segmenet containing whole audio
    if clen ==-1:
        i=0
        chunk = vid [:]
        chunk.export(f'{folder_path}/{i:05d}.mp3',format='mp3')
    else:
        for i in range(chunk_max):
            chunk = vid[i*chunk_length:(i+1)*chunk_length]
            chunk.export(f'{folder_path}/{i:05d}.mp3',format='mp3')
    return folder_path, chunk_length  # in sec

class TTS():
    def __init__(self):
        self.translator = Translator()
        self.translate = False
        print('INIT')

    def set_translate(self,new_val:bool=False):
        self.translate = new_val
        return 
    
    def get_translate_bool(self):
        return self.translate
                      
    def subtitles_video_with_display(self,filename,is_url:bool=False,url:str='https://youtu.be/oTN7xO6emU0',languag:str='en',display:bool=False,save:bool=False):
        if is_url:
            os.system(f'yt-dlp --verbose  --recode-video mp4 {url} -o {filename}')
            print('Finished downloading.')
        font = 0
        frame_cnt=-1
        folder_chunks = None
        chunk_len = None
        chunk_files = []
        subtitle_text = ''
        modelw = whisper.load_model("medium")
        subs_objs = []

        cap = cv2.VideoCapture(filename)
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if save:
            out = cv2.VideoWriter('sample_with_subs.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, size)

        print(f"{fps} frames per second")

 
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        subs = []
        chunk_audio =[]
        chunks = False
        if chunks:         
            folder_chunks, frames_interval = chunks_video(filename,clen=30)
            frames_interval = (frames_interval//1000)*fps

        else:
            folder_chunks, frames_interval = chunks_video(filename,clen=-1)

        chunk_files = [f'{folder_chunks}/{f}' for f in sorted(os.listdir(folder_chunks))]

        print(f'Chunk files: {chunk_files}')
        chunk_len = int(frames_interval//fps)
        print(f'Main-Frames interval:\t{frames_interval}')

        for chunk in chunk_files:
            chunk_audio.append(load_audio(chunk))
            
        # Read until video is completed
        subs_index=0
        srt_file_index = 0 #count total subtitle segments accross all file
        index_segment = 0
        all_results = []
        times = []
        subs_text = ''
        while(cap.isOpened()):
        # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                frame_cnt +=1
                #print(f'Modulo:\t{frame_cnt%frames_interval}')
                if frame_cnt%frames_interval==0:
                    # print('modulo 0')
                    # print(frame_cnt)

                    index_segment = frame_cnt//frames_interval
                    # print(f'Index segment: {index_segment}')
                    # print(f'Index segment: {chunk_audio[index_segment]}')

                    t0 = time.time()
                    result = modelw.transcribe(chunk_audio[index_segment],language=languag)
                    #print(result)
                    t1 = time.time() - t0
                    times.append(t1)
                    all_results.append(result)
                    for it in result['segments']:
                        srt_file_index+=1
                        subs.append( ( int(round(fps*it["start"])+index_segment*frames_interval)
                                        ,int(round(it["end"]*fps)+index_segment*frames_interval),
                                        it["text"]))
                        start_delta = timedelta(seconds=it["start"]+index_segment*chunk_len)
                        end_delta = timedelta(seconds=it["end"]+index_segment*chunk_len)
                        subs_objs.append(Subtitle(index=srt_file_index,
                                                    start=start_delta,
                                                    end=end_delta,
                                                    content=it['text']))
                        subs_text+=it['text']
                        subs_text+= '\n'
                        print(f'Start:\t{it["start"]}\tEnd:\t{it["end"]}Text:\t{it["text"]}')
                #try:
                if subs[subs_index][1]>frame_cnt:
                    subtitle_text = subs[subs_index][2]
                elif subs_index<len(subs)-1:
                    subs_index+=1
                    subtitle_text = subs[subs_index][2]
                else:
                    subtitle_text = ''
                #except IndexError as E:
                #    print(f'OIndex Error : {E}')




                if display:
                    try:
                        #print(subtitle_text)
                        pil_image = Image.fromarray(frame)
                        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 40, encoding="unic")
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((30, 30), f'{subtitle_text}\n{frame_cnt//fps}', font=font,fill="#0000FF")
                        frame = np.asarray(pil_image)

                        #frame = cv2.putText(frame, f'{subtitle_text}', org2, font,  fontScale, color2, thickness, cv2.LINE_AA)
                    except IndexError:
                        pass
                    cv2.imshow('Frame',frame)
                if save:
                    if not display:
                        pil_image = Image.fromarray(frame)
                        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 40, encoding="unic")
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((15, 15), f'{subtitle_text}', font=font,fill="#0000FF")
                        frame = np.asarray(pil_image)
                    out.write(frame)

                # Press Q on keyboard to  exit
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break

        # Break the loop

            else: 
                break
        
        # When everything done, release the video capture object
        all_test_subs = compose(subs_objs)
        with open(f"{filename[:-4]}.srt", "w") as f:
            f.write(all_test_subs)

        cap.release()
        text_file = open(f"{filename[:-4]}.txt", "w")
        n = text_file.write(subs_text)
        text_file.close()

        
        # Closes all the frames
        cv2.destroyAllWindows()
        if save: 
            out.release()

        if folder_chunks is not None:
            shutil.rmtree(folder_chunks)

############
    def subtitles_video(self,filename,is_url:bool=False,is_audio:bool=False,url:str='https://youtu.be/oTN7xO6emU0',languag:str='en',save_srt:bool=True,save_txt:bool=False,chunk_length:int=-1):
        if is_url:
            os.system(f'yt-dlp --verbose  --recode-video mp4 {url} -o {filename}')
            print('Finished downloading.')
        folder_chunks = None
        chunk_files = []
        modelw = whisper.load_model("medium")
        subs_objs = []
        chunk_audio =[]
        if chunk_length>0:         
            folder_chunks, frames_interval = chunks_video(filename,clen=chunk_length)
        else:
            folder_chunks, frames_interval = chunks_video(filename,clen=-1)

        chunk_files = [f'{folder_chunks}/{f}' for f in sorted(os.listdir(folder_chunks))]

        print(f'Chunk files: {chunk_files}')

        for chunk in chunk_files:
            chunk_audio.append(load_audio(chunk))
            
        # Read until video is completed
        srt_file_index = 0 #count total subtitle segments accross all file
        all_results = []
        times = []
        subs_text = ''
        for index_segment in range(len(chunk_audio)):
            t0 = time.time()
            result = modelw.transcribe(chunk_audio[index_segment],language=languag)
            #print(result)
            t1 = time.time() - t0
            times.append(t1)
            all_results.append(result)
            for it in result['segments']:
                srt_file_index+=1
                start_delta = timedelta(seconds=it["start"]+index_segment*chunk_length)
                end_delta = timedelta(seconds=it["end"]+index_segment*chunk_length)
                if self.translate:
                    text_trs = self.translator.translate(it['text']) 
                    subs_objs.append(Subtitle(index=srt_file_index,
                                            start=start_delta,
                                            end=end_delta,
                                            content=text_trs))
                else:
                    subs_objs.append(Subtitle(index=srt_file_index,
                                                start=start_delta,
                                                end=end_delta,
                                                content=it['text']))
                
                subs_text+=it['text']
                subs_text+= '\n'
                print(f'Start:\t{it["start"]}\tEnd:\t{it["end"]}Text:\t{it["text"]}')

        
        # When everything done, release the video capture object
        all_test_subs = compose(subs_objs)

        if save_srt:
            with open(f"{filename[:-4]}.srt", "w") as f:
                f.write(all_test_subs)
        
        if save_txt:
            with open(f"{filename[:-4]}.txt", "w") as text_file:
                n = text_file.write(subs_text)

        if folder_chunks is not None:
            shutil.rmtree(folder_chunks)
        
        return subs_text



###############################################
################ stand0alone functions ########
###############################################
            
def get_text_from_srt(filename:str ):
    
    srtfile = open(filename, "r", encoding="utf-8")
    text = srtfile.read()
    srtfile.close()
    subs = parse(text)
    all_subs = ''
    for s in subs:
        #start = s.start.seconds + s.start.microseconds/1000000
        #end  = s.end.seconds + s.end.microseconds/1000000
        all_subs+=s.content.replace('\n',' ').replace('<i>','').replace('</i>','')
        all_subs+='\n'
    return all_subs

def get_subs_from_srt(filename:str ):
    srtfile = open(filename, "r")
    text = srtfile.read()
    srtfile.close()
    subs = parse(text)
    subs_lst= []
    for s in subs:
        subs_lst.append(s)
    return subs_lst


def write_subs_to_video(video_fn:str,subs_fn:str,output_path:str='sample_with_subs_exp.mp4',font_sz:int=40,max_length:int=70):
    #if len(line)>upper_limit: insert /n at prev space
    subs = get_subs_from_srt(subs_fn)
    video = mp.VideoFileClip(video_fn)

    fps = int(round(video.fps))
    print(fps)
    audio = AudioSegment.from_file(video_fn,video_fn[-3:])
    audio.export('temp.mp3')

    frame_count = 0
    subs_index=0
    clips = []
    for frame in video.iter_frames():
        #print(frame_count)
        #plt.imshow(frame)
        #plt.show()
        try:
            s = subs[subs_index]

            start_s = int(round((s.start.seconds + (s.start.microseconds)/1000000)* fps)) 
            end_s = int(round((s.end.seconds + (s.end.microseconds)/1000000) * fps ))

            if end_s>frame_count and start_s<=frame_count:
                subtitle_text = s.content
            elif end_s==frame_count:
                subtitle_text = s.content
                subs_index+=1
            else:
                subtitle_text = ''

            if len(subtitle_text)>0:
                
                subtitle_text = break_chunk(subtitle_text,max_length=max_length)
                pil_image = Image.fromarray(frame)
                font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", font_sz, encoding="unic")
                draw = ImageDraw.Draw(pil_image)
                draw.text((30, 30), f'{subtitle_text}', font=font,fill="#FF00FF")
                frame = np.asarray(pil_image)
            clips.append(frame)
        except IndexError:
            pass
            #print(f'Out of subs.Lasts timestamp:{end_s}')

        frame_count+=1

    size = video.size
    temp_file = 'temp.mp4'
    out = cv2.VideoWriter(temp_file,cv2.VideoWriter_fourcc('m','p','4','v'), fps, size)
    for f in clips:
        out.write( cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    out.release()
    video_noa = mp.VideoFileClip(temp_file)
    audio_noa = mp.AudioFileClip('temp.mp3')


    video_noa = video_noa.set_audio(audio_noa)
    print(f'Saving file at :\t{output_path}')
    video_noa.write_videofile(output_path)

    os.remove('temp.mp3')
    os.remove('temp.mp4')

def break_chunk(chunk:str,max_length:int=50):
    positions = []
    mid = max_length//2
    min_diff = 1000000

    if len(chunk)>max_length:
        for i in range(len(chunk)):
            if chunk[i]=='_':
                positions.append(i)
        if len(positions)==0: #no space char found
            new_chunk = chunk[:mid] + '\n-' + chunk[mid:]
            return new_chunk
        else:
            for p in positions:
                if abs(mid-p)<min_diff:
                    min_diff=p
            new_chunk = chunk[:p] + '\n' + chunk[p:]
    else: 
        return chunk
    

def divide_text_into_sentence(text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    return sentences

def divide_text_into_sentence_batches(text, max_tokens_per_batch:int=128):
    """
    Divide a large text into batches of full sentences.
    
    Args:
    - text: The large text to be divided.
    - max_tokens_per_batch: The maximum number of tokens allowed in each batch.
    
    Returns:
    - A list of batches, where each batch contains full sentences.
    """
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)
    
    # Initialize variables
    batches = []
    current_batch = []
    current_batch_length = 0
    
    # Iterate through sentences and create batches
    for sentence in sentences:
        sentence_length = len(nltk.word_tokenize(sentence))
        if current_batch_length + sentence_length <= max_tokens_per_batch:
            current_batch.append(sentence)
            current_batch_length += sentence_length
        else:
            batches.append(' '.join(current_batch))
            current_batch = [sentence]
            current_batch_length = sentence_length

    
    # Add the last batch
    if current_batch:
        batches.append(' '.join(current_batch))
    
    return batches



class Translator():
    def __init__(self, model_name:str = "facebook/nllb-200-distilled-600M"):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.detection_model = fasttext.load_model("lid218e.bin")
        self.source_language = None
        self.target_language =  None
        with open('flora200.json') as f:
            self.language_tags = json.load(f)

    def get_tag_languages(self,print_list:bool=False):
        #maybe find code for language as well
        if print_list:
            print('Language\t\t:\tCode')
            for k,i in self.language_tags.items():
                print(f'{k}\t\t:\t{i}')

        return self.language_tags

    def set_source_language(self,language:str):
        #if language is valid here
        if language in list(self.language_tags.values()):
            self.source_language = language
        else:
            print(f'{language} not found. Please insert a valid language')

        return
    
    def get_target_language(self,):
        return self.target_language
    
    def set_target_language(self,language:str):
        #if language is valid here
        if language in list(self.language_tags.values()):
            self.target_language = language
        else:
            print(f'{language} not found. Please insert a valid language')

        return
    
    def get_source_language(self,):
        return self.source_language

    def translate_sentence(self,text:str):

        translator = pipeline('translation', model=self.model,
                            tokenizer=self.tokenizer,
                            src_lang=self.source_language,
                            tgt_lang=self.target_language)
        output = translator(text, max_length=1024)
        prompt = output[0]['translation_text']
        #print(prompt)
        return prompt
    
    def translate(self,text):

        if self.source_language==None:
            self.source_language = self.detect_language(text)

        if self.target_language == None:
            print('Please set the target language.')
            return ''
        batches = divide_text_into_sentence(text)
        full_text = ''
        for b in batches:
            translation_b = self.translate_sentence(b)
            full_text+=translation_b 

        return full_text


    def detect_language(self,text:str):
        print('Detecting source language')
        predictions = self.detection_model.predict(text, k=1)
        return predictions[0][0].replace('__label__', '')


class Summarizer():
    def __init__(self,model_name:str='facebook/bart-large-cnn'):
        self.model_name = model_name
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.max_length=150
        self.min_length = 40
        self.length_penalty = 2.0
        self.num_beams = 4
        self.early_stop = True

        ###check tokenizer's  enoder max length

    def summarize(self,input_text:str):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)

        summary_ids = self.model.generate(input_ids, 
                                          max_length=self.max_length, 
                                          min_length=self.min_length, 
                                          length_penalty=self.length_penalty, 
                                          num_beams=self.num_beams,
                                          early_stopping=self.early_stop)

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    
class SummarizerQwen():
    def __init__(self,qwen_model_name:str="Qwen/Qwen1.5-0.5B-Chat",device:str = "cuda"):
        self.model_name = qwen_model_name
        self.model = AutoModelForCausalLM.from_pretrained(
                                                qwen_model_name,
                                                torch_dtype="auto",
                                                ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)

        self.device = device
        self.max_length=150
        self.min_length = 40
        self.length_penalty = 2.0
        self.num_beams = 4
        self.early_stop = True

        ###check tokenizer's  enoder max length

    def summarize(self,task:str, 
                  input_text:str,
                  prompt_sys:str = "You are a helpful assistant.",
                  max_new_length_tokens:int=256,
                  save_to_file:bool=True,
                  output_filename:str="out.txt"):
        prompt = f"{task} : {input_text}"
        messages = [
            {"role": "system", "content": prompt_sys},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_length_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if save_to_file:
            with open(output_filename,"w") as of:
                of.write(response)
        return response