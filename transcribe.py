import argparse
from utils import TTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', metavar='str', nargs=1, default=['sample.mp4'], help='Model name')
    parser.add_argument('--is_url', type=bool, nargs=1, default=False, help='Indicates if source is local or not')
    parser.add_argument('--url', metavar='str', nargs=1, default=['https://youtu.be/oTN7xO6emU0'], help='Link to download the video from if URL.')
    parser.add_argument('--transcription_language', metavar='str', nargs=1, default=['en'], help='Language to transcribe.')

    parser.add_argument('--device', metavar='str', nargs=1, default=['cuda'], help='cpu or cuda')

    parser.add_argument('--translate', type=bool, nargs=1, default=False, help='Translate or keep original language')
    parser.add_argument('--source_language', metavar='str', nargs=1, default=['unknown'], help='Language to translate from.')
    parser.add_argument('--target_language', metavar='str', nargs=1, default=['tur_Latn'], help='Language to translate to')


    args = parser.parse_args()

    video_filename = args.video_file[0]
    print(f'Video: \t{video_filename}')
    is_url= args.is_url
    print(f'Is url: \t{is_url}')
    url = args.url[0]
    print(f'URL:\t{url}')
    transcription_language = args.transcription_language[0]
    print(f'Transcription language: \t{transcription_language}')

    device = args.device[0]
    print(f'Device: \t{device}')

    translate = args.translate
    print(f'Translate: \t{translate}')
    source_language = args.source_language[0]
    print(f'source language: \t{source_language}')
    target_language = args.target_language[0]
    print(f'target language: \t{target_language}')

    stt = TTS()
    stt.translate = translate
    if translate:
        if source_language!='unknown':
            stt.translator.set_source_language (source_language)
        stt.translator.set_target_language (target_language)

    stt.subtitles_video(filename=video_filename,
                        is_url=url,
                        url=url,
                        language=transcription_language)