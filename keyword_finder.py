"""
Video Keyword Finder
==================

A Python script that finds and timestamps specific keywords within video files.
It transcribes the audio using Whisper AI and finds all occurrences of specified keywords
with their timestamps and surrounding context. Supports both local video files and YouTube URLs.

Requirements
-----------
- Python 3.8 or higher
- ffmpeg (must be installed and accessible in system PATH)
- GPU recommended but not required

Installation
-----------
1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Unix/MacOS
   ```

2. Install required packages:
   ```
   pip install yt-dlp whisper-timestamped torch ffmpeg-python
   ```

3. Download the Whisper model (happens automatically on first run)

Usage Examples
-------------
1. Search keywords in a local video file:
   ```
   python keyword_finder.py path/to/video.mp4 -k search_term
   ```

2. Search with multiple keywords and time range:
   ```
   python keyword_finder.py video.mp4 -k keyword1 keyword2 -b 0:00 -e 10:00
   ```

3. Search in a YouTube video:
   ```
   python keyword_finder.py https://www.youtube.com/watch?v=VIDEO_ID -k word1 word2
   ```

Output
------
- Generates 'timestamps.txt' with all keyword occurrences and their context
- Creates a detailed log file 'keyword_finder.log'
- Shows progress in console during processing

Note: Processing time depends on video length and system capabilities.
      A 1-hour video typically takes 15-30 minutes to process.
"""

import argparse
import whisper_timestamped as whisper
import datetime
import os
import tempfile
import yt_dlp
import re
import logging
import math
import subprocess
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keyword_finder.log'),
        logging.StreamHandler()
    ]
)

def parse_time(time_str):
    """Convert time string (HH:MM:SS) to seconds"""
    if time_str is None:
        return None
    try:
        time_parts = list(map(int, time_str.split(':')))
        if len(time_parts) == 3:  # HH:MM:SS
            return time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
        elif len(time_parts) == 2:  # MM:SS
            return time_parts[0] * 60 + time_parts[1]
        else:
            return int(time_str)  # Just seconds
    except:
        raise ValueError("Time must be in HH:MM:SS, MM:SS, or seconds format")

def format_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.timedelta(seconds=int(seconds)))

def is_youtube_url(url):
    """Check if the provided string is a YouTube URL"""
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
    return bool(re.match(youtube_regex, url))

def download_youtube_video(url):
    """Download YouTube video and return path to temporary file"""
    temp_dir = tempfile.gettempdir()
    temp_file = os.path.join(temp_dir, 'youtube_video.mp4')
    
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': temp_file,
        'quiet': False,
        'progress_hooks': [lambda d: logging.info(f"Download progress: {d.get('status', 'unknown')}")],
        'socket_timeout': 30,
    }
    
    try:
        logging.info(f"Starting download of YouTube video: {url}")
        print("Downloading video... This may take a few minutes.")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            logging.info(f"Video info extracted: {info.get('title', 'Unknown title')}")
        
        if os.path.exists(temp_file):
            file_size = os.path.getsize(temp_file)
            logging.info(f"Download complete. File size: {file_size / (1024*1024):.2f} MB")
            return temp_file
        else:
            raise Exception("Download completed but file not found")
    except Exception as e:
        logging.error(f"Error downloading YouTube video: {str(e)}")
        raise

def get_video_duration(video_path):
    """Get video duration using ffprobe"""
    cmd = [
        'ffprobe', 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        video_path
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting video duration: {str(e)}")
        raise

def split_video(video_path, segment_duration=120):
    """Split video into segments using ffmpeg"""
    try:
        temp_dir = tempfile.gettempdir()
        segment_pattern = os.path.join(temp_dir, 'segment_%03d.mp4')
        
        # Remove any existing segments
        for old_segment in glob.glob(os.path.join(temp_dir, 'segment_*.mp4')):
            try:
                os.remove(old_segment)
            except:
                pass
        
        # Split video into segments
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-f', 'segment',
            '-segment_time', str(segment_duration),
            '-c', 'copy',
            '-reset_timestamps', '1',
            segment_pattern
        ]
        
        logging.info("Splitting video into segments...")
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Get list of generated segments
        segments = sorted(glob.glob(os.path.join(temp_dir, 'segment_*.mp4')))
        logging.info(f"Created {len(segments)} segments")
        
        return segments
        
    except Exception as e:
        logging.error(f"Error splitting video: {str(e)}")
        raise

def process_segments(segments, keywords):
    """Process each segment sequentially"""
    results = {keyword: [] for keyword in keywords}
    
    # Load whisper model once
    logging.info("Loading Whisper model...")
    model = whisper.load_model("base")
    
    for i, segment_path in enumerate(segments):
        try:
            segment_num = int(re.search(r'segment_(\d+)', segment_path).group(1))
            start_time = segment_num * 120  # Each segment is 120 seconds
            
            logging.info(f"Processing segment {i+1}/{len(segments)} (starting at {format_time(start_time)})")
            
            # Transcribe segment
            audio = whisper.load_audio(segment_path)
            transcription = whisper.transcribe(model, audio)
            
            # Process results
            for segment in transcription['segments']:
                text = segment['text'].lower()
                timestamp = segment['start'] + start_time  # Adjust timestamp relative to full video
                
                for keyword in keywords:
                    if keyword.lower() in text:
                        results[keyword].append({
                            'timestamp': timestamp,
                            'text': segment['text']
                        })
                        logging.info(f"Found keyword '{keyword}' at {timestamp:.2f}s: {segment['text']}")
            
        except Exception as e:
            logging.error(f"Error processing segment {segment_path}: {str(e)}")
            continue
            
        finally:
            # Clean up segment file
            try:
                os.remove(segment_path)
            except:
                pass
    
    return results

def find_keywords_in_video(video_path, keywords, begin_time=None, end_time=None):
    """Find timestamps for keywords in video transcription"""
    try:
        logging.info(f"Processing video: {video_path}")
        logging.info(f"Searching for keywords: {keywords}")
        
        # Convert keywords string to list and clean up
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(',')]
        
        # Get video duration
        duration = get_video_duration(video_path)
        logging.info(f"Video duration: {duration:.2f} seconds")
        
        # Set time bounds
        start = begin_time if begin_time is not None else 0
        end = min(end_time if end_time is not None else duration, duration)
        
        # Split video into segments
        segment_duration = 120  # 2 minutes per segment
        segments = split_video(video_path, segment_duration)
        
        # Process segments sequentially
        results = process_segments(segments, keywords)
        
        return results
        
    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        raise

def main():
    try:
        parser = argparse.ArgumentParser(description='Find keywords in video and output their timestamps')
        parser.add_argument('video', help='Path to video file or YouTube URL')
        parser.add_argument('--begin', help='Begin time (HH:MM:SS, MM:SS, or seconds)', default=None)
        parser.add_argument('--end', help='End time (HH:MM:SS, MM:SS, or seconds)', default=None)
        parser.add_argument('--keywords', required=True, help='Comma-separated list of keywords to find')
        parser.add_argument('--output', required=True, help='Output file path')
        
        args = parser.parse_args()
        
        # Handle YouTube URLs
        video_path = args.video
        temp_file = None
        
        if is_youtube_url(args.video):
            print("Detected YouTube URL. Downloading video...")
            video_path = download_youtube_video(args.video)
            temp_file = video_path
            print("Download complete. Processing video...")
            
        # Check if video file exists (for local files)
        elif not os.path.exists(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
            
        try:
            # Convert time arguments to seconds
            begin_time = parse_time(args.begin) if args.begin else None
            end_time = parse_time(args.end) if args.end else None
            
            # Validate time range
            if begin_time is not None and end_time is not None and begin_time >= end_time:
                raise ValueError("End time must be greater than begin time")
                
            # Find keywords
            results = find_keywords_in_video(video_path, args.keywords, begin_time, end_time)
            
            # Check if any matches were found
            total_matches = sum(len(matches) for matches in results.values())
            
            # Write results to file
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Keyword Timestamps for: {args.video}\n")
                f.write(f"Time range: {format_time(begin_time) if begin_time else 'start'} to {format_time(end_time) if end_time else 'end'}\n")
                f.write(f"Total matches found: {total_matches}\n\n")
                
                if total_matches == 0:
                    f.write("No matches found for any keywords.\n")
                
                for keyword, matches in results.items():
                    f.write(f"\nKeyword: {keyword}\n")
                    f.write("-" * 40 + "\n")
                    if not matches:
                        f.write("No matches found\n")
                    for match in matches:
                        f.write(f"[{format_time(match['timestamp'])}] {match['text']}\n")
                    f.write("\n")
                    
            print(f"Results written to {args.output}")
            if total_matches == 0:
                print("Warning: No matches found for any keywords")
                
        finally:
            # Clean up temporary file if it was created
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
