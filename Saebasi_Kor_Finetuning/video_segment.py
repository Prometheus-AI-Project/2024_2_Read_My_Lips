import os
import cv2  # OpenCV를 사용하여 비디오 처리
from pydub import AudioSegment  # Pydub를 사용하여 오디오 처리
import whisper  # Whisper를 사용하여 음성 인식
import subprocess  # subprocess 모듈을 사용하여 ffmpeg 호출

def extract_audio(video_path, audio_path):
    """비디오에서 오디오를 추출합니다."""
    video = cv2.VideoCapture(video_path)
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="wav")

def transcribe_audio(audio_path, word_timestamp=True):
    """오디오 파일을 텍스트로 변환하고 단어의 절대적 타임스탬프를 반환합니다."""
    model = whisper.load_model("large")  # Whisper large 모델 로드
    result = model.transcribe(audio_path, language='ko', task='transcribe', word_timestamps=word_timestamp)  # word_timestamps 매개변수 사용
    
    # 결과 확인
    timestamps = []  # 빈 리스트로 초기화
    if 'segments' in result:
        for segment in result['segments']:
            if 'words' in segment:
                for word_info in segment['words']:
                    # 절대적 타임스탬프 추가
                    timestamps.append({
                        'word': word_info['word'],
                        'start': word_info['start'] + segment['start'],  # 절대적 시작 시간
                        'end': word_info['end'] + segment['start']       # 절대적 끝 시간
                    })

    return result['text'], timestamps  # 텍스트와 타임스탬프 반환

def create_segments(transcription, words_per_segment):
    """전사를 단어 단위로 나누어 세그먼트를 생성합니다."""
    words = transcription.split()  # transcription이 텍스트여야 함
    segments = []
    for i in range(0, len(words), words_per_segment):
        segment = words[i:i + words_per_segment]
        segments.append(' '.join(segment))
    return segments

def slice_video_and_audio(video_path, audio_path, segments, output_folder, fps=30):
    """비디오와 오디오를 세그먼트에 따라 슬라이스합니다."""
    video = cv2.VideoCapture(video_path)
    audio = AudioSegment.from_file(audio_path)
    
    duration_per_segment = video.get(cv2.CAP_PROP_FRAME_COUNT) / fps / len(segments)
    
    # 비디오, 오디오, 전사 텍스트를 저장할 하위 폴더 생성
    video_output_folder = os.path.join(output_folder, "video_segments")
    audio_output_folder = os.path.join(output_folder, "audio_segments")
    transcription_output_folder = os.path.join(output_folder, "transcriptions")
    
    os.makedirs(video_output_folder, exist_ok=True)
    os.makedirs(audio_output_folder, exist_ok=True)
    os.makedirs(transcription_output_folder, exist_ok=True)
    
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    for idx, segment in enumerate(segments):
        start_time = idx * duration_per_segment * 1000  # milliseconds
        end_time = start_time + duration_per_segment * 1000
        
        # 오디오 슬라이스
        audio_segment = audio[start_time:end_time]
        audio_segment_path = os.path.join(audio_output_folder, f"{video_basename}_{idx + 1}.wav")
        audio_segment.export(audio_segment_path, format="wav")
        
        # 전사 텍스트 저장
        transcription, timestamps = transcribe_audio(audio_segment_path)  # 오디오를 전사
        with open(os.path.join(transcription_output_folder, f"{video_basename}_{idx + 1}.txt"), 'w', encoding='utf-8') as f:
            f.write(transcription)
            f.write("\nTimestamps:\n")
            for word_info in timestamps:
                f.write(f"{word_info['word']} {word_info['start']} {word_info['end']}\n")  # 단어와 절대적 타임스탬프 저장
        
        # 비디오 슬라이스 및 결합
        video_segment_path = os.path.join(video_output_folder, f"{video_basename}_{idx + 1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_segment_path, fourcc, fps, (int(video.get(3)), int(video.get(4))))
        
        # 비디오 프레임을 읽고 처리
        while True:
            ret, frame = video.read()
            current_time = video.get(cv2.CAP_PROP_POS_MSEC)
            if not ret or current_time > end_time:
                break
            if current_time >= start_time:
                out.write(frame)
        
        out.release()
        
        # 비디오와 오디오 결합
        combined_output_folder = os.path.join(output_folder, "combined_segments")
        os.makedirs(combined_output_folder, exist_ok=True)
        combined_segment_path = os.path.join(combined_output_folder, f"{video_basename}_{idx + 1}.mp4")
        
        combine_video_and_audio(video_segment_path, audio_segment_path, combined_segment_path)  # 비디오와 오디오 결합
        
        # 메모리 해제
        del audio_segment
        del transcription
        del timestamps

    return video_output_folder, audio_output_folder  # 비디오 및 오디오 출력 폴더 반환

def combine_video_and_audio(video_path, audio_path, output_path):
    """비디오와 오디오 파일을 결합합니다."""
    command = [
        'ffmpeg',
        '-i', video_path,  # 입력 비디오 파일
        '-i', audio_path,   # 입력 오디오 파일
        '-c:v', 'copy',     # 비디오 코덱 복사
        '-c:a', 'aac',      # 오디오 코덱 설정
        '-strict', 'experimental',  # 실험적 기능 사용
        output_path         # 출력 파일 경로
    ]
    
    subprocess.run(command, check=True)  # ffmpeg 명령어 실행

def create_lrs2_dataset(input_folder, output_folder, words_per_segment=3):
    """LRS2 데이터셋 형식으로 변환합니다."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 입력 폴더 내 모든 비디오 파일 처리
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]  # 지원하는 비디오 형식

    # 숫자 순서로 정렬하는 함수
    def sort_key(file_name):
        # 파일 이름에서 숫자 부분을 추출하여 정수로 변환
        return int(file_name.split('_')[-1].split('.')[0])

    # 파일 이름을 숫자 순서로 정렬
    video_files.sort(key=sort_key)

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        video_output_folder = os.path.join(output_folder, os.path.splitext(video_file)[0])  # 비디오 파일명으로 폴더 생성
        os.makedirs(video_output_folder, exist_ok=True)

        audio_path = os.path.join(video_output_folder, "extracted_audio.wav")
        extract_audio(video_path, audio_path)

        # 전사 텍스트를 생성하기 전에 오디오를 세그먼트로 나누기
        transcription, _ = transcribe_audio(audio_path)  # 전사된 텍스트를 가져오기
        segments = create_segments(transcription, words_per_segment)  # 전사된 텍스트를 세그먼트로 나누기
        
        # 비디오와 오디오 슬라이스
        audio_output_folder = os.path.join(video_output_folder, "audio_segments")
        video_output_segments_folder = os.path.join(video_output_folder, "video_segments")
        combined_output_folder = os.path.join(video_output_folder, "combined_segments")
        os.makedirs(audio_output_folder, exist_ok=True)
        os.makedirs(video_output_segments_folder, exist_ok=True)
        os.makedirs(combined_output_folder, exist_ok=True)

        video_output_folder, audio_output_folder = slice_video_and_audio(video_path, audio_path, segments, video_output_folder)

        print(f"{video_file}에 대한 세그먼트 생성 완료")

# 사용 예시
input_directory = "/home/ysoh20/finetuning/input"  # 입력 비디오 파일이 있는 디렉토리
output_directory = "/home/ysoh20/finetuning/output"  # 출력 폴더
create_lrs2_dataset(input_directory, output_directory, words_per_segment=3)