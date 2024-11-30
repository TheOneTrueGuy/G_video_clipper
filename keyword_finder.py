import argparse
import whisper_timestamped as whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import datetime
import os
import tempfile
import yt_dlp
import re

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
        'quiet': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return temp_file
    except Exception as e:
        raise Exception(f"Error downloading YouTube video: {str(e)}")

def find_keywords_in_video(video_path, keywords, begin_time=None, end_time=None):
    """Find timestamps for keywords in video transcription"""
    try:
        # Load the model and transcribe
        model = whisper.load_model("base")
        audio = whisper.load_audio(video_path)
        
        # Get video duration if end_time is None
        if end_time is None:
            try:
                with VideoFileClip(video_path) as video:
                    end_time = video.duration
            except Exception as e:
                print(f"Warning: Could not get video duration: {str(e)}")
                end_time = float('inf')  # Use infinite duration as fallback

        # Convert begin_time to 0 if None
        begin_time = begin_time or 0
        
        # Transcribe the audio
        try:
            result = whisper.transcribe(model, audio, language="en")
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return {}
        
        # Process keywords and create results dictionary
        keyword_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]  # Skip empty keywords
        if not keyword_list:
            print("Warning: No valid keywords provided")
            return {}
            
        results = {keyword: [] for keyword in keyword_list}
        
        # Search through segments
        for segment in result.get("segments", []):
            # Skip if segment is outside our time range
            if segment["start"] < begin_time or segment["end"] > end_time:
                continue
                
            text = segment["text"].lower()
            for keyword in keyword_list:
                if keyword in text:
                    results[keyword].append({
                        "timestamp": segment["start"],
                        "text": segment["text"].strip()
                    })
        
        return results
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return {}

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
