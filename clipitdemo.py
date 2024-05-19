# python -m venv vidclipr
# .\vidclipr\Scripts\Activate

# pip install -U openai-whisper # mock pytube
# pip install -U git+https://github.com/linto-ai/whisper-timestamped

import os
import datetime as dt
#from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import json
import whisper_timestamped as whisper
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
from google.colab import output

parser = argparse.ArgumentParser(description='Process video file.')
parser.add_argument('-v', '--video', dest='video_file', required=True, help='Path to the video file')
args = parser.parse_args()
video_file_path = args.video_file
name=video_file_path

#make us a directory to put our clips in, will be used to zip later
outdir = dt.datetime.now().strftime("%Y%m%d%H%M")
# if outdir exists add 4 random digits to it:
if os.path.exists(outdir):
    # Generate 4 random digits
    random_digits = str(random.randint(1000, 9999))

    # Create a new output directory with the random digits appended
    new_outdir = outdir + random_digits

    os.mkdir(new_outdir)
    outdir=new_outdir
    print("Created new output directory:", new_outdir)
else:
    os.system(f"mkdir {outdir}")
print("date time now:" + outdir)

model = whisper.load_model("base")

def generate_timestamps(vidname):
  audio = whisper.load_audio(vidname)
  result = whisper.transcribe(model, audio, language="en")
  return result

result=generate_timestamps(name) #now result becomes a global list and only needs to be loaded once

def get_segment_info(data):
    new_list = []
    for segment in data.get("segments", []):
        if "id" in segment and "start" in segment and "end" in segment and "text" in segment:
            new_item = {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            }
            new_list.append(new_item)
    return new_list
# Print the result
print(json.dumps(get_segment_info(result) , indent=2))

def combine_entries(entries):
    combined_entries = []
    current_entry = None
    total_duration = 0

    for entry in entries:
        entry_duration = entry["end"] - entry["start"]

        # If adding the current entry exceeds 30 seconds, create a new combined entry
        if total_duration + entry_duration > 30:
            if current_entry:
                current_entry["end"] = entry["end"]
                combined_entries.append(current_entry)

            # Reset for a new combined entry
            current_entry = {
                "start": entry["start"],
                "end": entry["end"],
                "text": entry["text"]
            }
            total_duration = entry_duration
        else:
            # Add to the current combined entry
            if current_entry:
                current_entry["end"] = entry["end"]
                current_entry["text"] += " " + entry["text"]
                total_duration += entry_duration
            else:
                # If no current entry, start a new one
                current_entry = {
                    "start": entry["start"],
                    "end": entry["end"],
                    "text": entry["text"]
                }
                total_duration = entry_duration

    # Add the last combined entry if it exists
    if current_entry:
        combined_entries.append(current_entry)

    return combined_entries


combined_entries = combine_entries(get_segment_info(result))
print(json.dumps(combined_entries , indent=2))


def extract_video_segment(input_video, output_video, start_time, end_time):
    video_clip = VideoFileClip(input_video).subclip(start_time, end_time)
    video_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")
    video_clip.close()

def save_segments(outdir, name):
    print("segmenting")
    # result = generate_timestamps(name)

    segments = combined_entries #get_segment_info(result) 
    input_video = name  # load_video(name)
    total=len(segments)
    for i, segment in enumerate(segments):
        start_time = segment['start']
        end_time = segment['end']
        output_video_file = f'{outdir}/output_segment_{i + 1}.mp4'
        extract_video_segment(input_video, output_video_file, start_time, end_time)

print(dt.datetime.now().strftime("%Y%m%d%H%M"))

print(dt.datetime.now().strftime("%Y%m%d%H%M"))
save_segments(outdir, name)

print(dt.datetime.now().strftime("%Y%m%d%H%M"))

scribeout=open(f"{outdir}/transcript.txt", "w")
scribeout.write(json.dumps(combined_entries, indent = 2, ensure_ascii = False))
scribeout.close()

filename, extension = os.path.splitext(name)
os.system(f"zip -r {filename}.zip  {outdir}")
