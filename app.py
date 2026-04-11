from flask import Flask, render_template, request, send_file, flash, redirect, url_for, jsonify
import os
import speech_recognition as sr
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import tempfile
import shutil
from werkzeug.utils import secure_filename
import logging
import time
import gc
import threading
import uuid
from faster_whisper import WhisperModel
import json
import multiprocessing
from proglog import ProgressBarLogger
import time

# Detect serverless environment - check for Vercel/Lambda env vars only
IS_SERVERLESS = os.environ.get('VERCEL') == '1' or os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

# Configure ImageMagick and FFmpeg for MoviePy
if 'IMAGEMAGICK_BINARY' not in os.environ:
    if os.name == 'nt':  # Windows
        possible_paths = [
            r'C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe',
            r'C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe',
            r'C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick.exe'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ['IMAGEMAGICK_BINARY'] = path
                break
    else:
        # Linux/Serverless - use system ImageMagick
        os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'

# Configure FFmpeg for serverless (Vercel/AWS Lambda)
if IS_SERVERLESS:
    # Check for common FFmpeg locations
    ffmpeg_paths = ['/usr/bin/ffmpeg', '/usr/local/bin/ffmpeg', '/opt/ffmpeg/ffmpeg', 'ffmpeg']
    for path in ffmpeg_paths:
        if os.path.exists(path) or os.system(f'which {path} > /dev/null 2>&1') == 0:
            os.environ['FFMPEG_BINARY'] = path if os.path.exists(path) else path
            break
    # Update MoviePy config
    try:
        from moviepy.config import change_settings
        change_settings({"FFMPEG_BINARY": os.environ.get('FFMPEG_BINARY', 'ffmpeg')})
        change_settings({"IMAGEMAGICK_BINARY": os.environ.get('IMAGEMAGICK_BINARY', 'convert')})
    except:
        pass


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

@app.route('/favicon.ico')
def favicon():
    return '', 204

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Werkzeug request logs (stops the INFO:werkzeug GET/POST messages)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Create necessary directories - use /tmp in serverless environments
if IS_SERVERLESS:
    UPLOAD_FOLDER = '/tmp/uploads'
    OUTPUT_FOLDER = '/tmp/output'
else:
    UPLOAD_FOLDER = 'uploads'
    OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global dictionary to track progress (in-memory only for serverless compatibility)
progress_tracker = {}
PROGRESS_FILE = '/tmp/progress.json' if IS_SERVERLESS else 'progress.json'

def save_progress():
    """Progress tracking - in-memory with optional file backup"""
    if not IS_SERVERLESS:
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress_tracker, f)
        except:
            pass

def load_progress():
    """Progress tracking - load from file if exists"""
    if not IS_SERVERLESS and os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                global progress_tracker
                progress_tracker = json.load(f)
        except:
            pass

# Resume not supported with in-memory only progress tracking

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hex_to_rgb(hex_color):
    """Convert hex color string (#RRGGBB) to RGB tuple (R, G, B)"""
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (255, 255, 255) # Default to white on error

def extract_audio_from_video(video_path):
    """Extract audio from video file"""
    try:
        video = VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '_audio.wav').replace('.avi', '_audio.wav').replace('.mov', '_audio.wav')
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {str(e)}")
        raise

def transcribe_audio_with_timestamps(audio_path):
    """Transcribe audio to text using Whisper with timestamps"""
    try:
        logger.info(f"Starting Whisper transcription for: {audio_path}")
        
        # Load faster-whisper model
        model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully")
        
        # Transcribe with word-level timestamps
        segments, info = model.transcribe(audio_path, word_timestamps=True)
        logger.info(f"Whisper transcription completed - language: {info.language}")
        
        # Extract word-level timestamps
        words = []
        full_text_parts = []
        for segment in segments:
            full_text_parts.append(segment.text)
            if segment.words:
                for word in segment.words:
                    words.append({
                        "text": word.word.strip(),
                        "start": word.start,
                        "end": word.end
                    })
        
        # Create full text
        full_text = " ".join(full_text_parts).strip()
        logger.info(f"Transcription successful: {full_text[:100]}...")
        logger.info(f"Generated {len(words)} word timestamps")
        
        return full_text, words
        
    except Exception as e:
        logger.error(f"Error transcribing audio with Whisper: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        # Fallback to simple speech recognition
        return transcribe_audio_fallback(audio_path), []

def transcribe_audio_fallback(audio_path):
    """Fallback transcribe using speech recognition"""
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
        
        # Use Google's speech recognition
        text = recognizer.recognize_google(audio)
        logger.info(f"Fallback transcription successful: {text[:100]}...")
        return text, []
    except sr.UnknownValueError:
        logger.error("Speech recognition could not understand audio")
        return "Could not understand audio", []
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {str(e)}")
        return f"Speech recognition service error: {str(e)}", []
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return "Error transcribing audio", []

def group_words_into_phrases(caption_timestamps, max_words_per_phrase=3):
    """Group individual words into phrases to reduce clip count and speed up rendering"""
    if not caption_timestamps or len(caption_timestamps) <= 3:
        return caption_timestamps
    
    phrases = []
    current_phrase_words = []
    
    for i, word_data in enumerate(caption_timestamps):
        current_phrase_words.append(word_data)
        
        # Create phrase when we hit max words or at end
        if len(current_phrase_words) >= max_words_per_phrase or i == len(caption_timestamps) - 1:
            phrase_text = ' '.join(w['text'] for w in current_phrase_words)
            phrase_start = current_phrase_words[0]['start']
            phrase_end = current_phrase_words[-1]['end']
            
            phrases.append({
                'text': phrase_text,
                'start': phrase_start,
                'end': phrase_end
            })
            current_phrase_words = []
    
    return phrases

def create_caption_preview(video_path, caption_text, job_id):
    """Create a preview image showing where captions will appear"""
    try:
        progress_tracker[job_id] = {'step': 'Creating preview', 'progress': 50}
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Get first frame
        first_frame = video.get_frame(0)
        
        # Create preview showing speech timing approach
        words = caption_text.split()
        preview_text = f"Words appear at 1s, spread evenly across video"
        
        txt_clip = TextClip(preview_text, 
                          fontsize=18,
                          color='yellow',
                          font='Arial',
                          stroke_color='red',
                          stroke_width=2)
        
        # Position text higher from bottom, centered
        txt_clip = txt_clip.set_position(('center', video.h - 100))
        
        # Create composite with first frame
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        frame_clip = ImageSequenceClip([first_frame], fps=1)
        preview = CompositeVideoClip([frame_clip, txt_clip])
        
        # Save preview image in uploads folder
        video_basename = os.path.basename(video_path)
        preview_filename = video_basename.replace('.mp4', '_preview.png').replace('.avi', '_preview.png')
        preview_path = os.path.join(UPLOAD_FOLDER, preview_filename)
        preview.save_frame(preview_path, t=0)
        
        video.close()
        txt_clip.close()
        frame_clip.close()
        preview.close()
        
        return preview_path
        
    except Exception as e:
        logger.error(f"Error creating preview: {str(e)}")
        return None

class MoviePyProgressLogger(ProgressBarLogger):
    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id
        self.last_update_time = 0

    def bars_callback(self, bar, attr, value, old_value=None):
        # In proglog 0.1.12, 'bar' is the bar NAME (string), not the bar object
        try:
            bar_dict = self.bars.get(bar)
            if not bar_dict:
                return

            if attr == 'index':
                current_time = time.time()
                total = bar_dict.get('total', 0)
                
                if current_time - self.last_update_time > 0.5 or value == total:
                    if total > 0:
                        percentage_done = int((value / total) * 100)
                        # Map 0-100% render to 80-99% total
                        overall_percentage = 80 + int((value / total) * 19)
                        
                        if self.job_id in progress_tracker:
                            progress_tracker[self.job_id]['progress'] = overall_percentage
                            progress_tracker[self.job_id]['step'] = f"Rendering video... {percentage_done}%"
                            self.last_update_time = current_time
        except Exception as e:
            # Never let progress updates crash the actual rendering
            pass

def add_captions_to_video(video_path, caption_text, caption_timestamps, output_path, job_id, caption_styles=None):
    """Add captions to video using MoviePy with professional styling"""
    try:
        logger.info(f"Adding captions to video: {video_path}")
        
        # Load video
        video = VideoFileClip(video_path)
        caption_clips = []
        
        # Get caption styles with defaults
        font_size_map = {
            'xsmall': 10,
            'small': 14,
            'medium': 18, 
            'large': 22,
            'xlarge': 26,
            'xxlarge': 32
        }
        font_family_map = {
            'inter': 'Arial',
            'arial': 'Arial',
            'helvetica': 'Arial',
            'times': 'Times-New-Roman',
            'courier': 'Courier-New',
            'georgia': 'Georgia',
            'verdana': 'Verdana'
        }
        color_map = {
            'white': 'white',
            'yellow': 'yellow',
            'cyan': 'cyan',
            'lime': 'lime',
            'red': 'red',
            'black': 'black'
        }
        bg_map = {
            'semi-black': (0, 0, 0, 0.8),
            'black': (0, 0, 0, 0.95),
            'none': None,
            'semi-white': (255, 255, 255, 0.8)
        }
        
        # Apply styles with defaults
        if caption_styles is None:
            caption_styles = {}
            
        # Get base settings
        f_size = font_size_map.get(caption_styles.get('fontSize', 'medium'), 18)
        f_family_base = font_family_map.get(caption_styles.get('fontFamily', 'inter'), 'Arial')
        f_style = caption_styles.get('fontStyle', 'normal')
        
        # Resolve full font name for ImageMagick (Bold/Italic support)
        if f_style == 'bold':
            font_full_name = f"{f_family_base}-Bold"
        elif f_style == 'italic':
            # Handle fonts that use 'Oblique' instead of 'Italic'
            if f_family_base in ['Helvetica', 'Courier-New']:
                font_full_name = f"{f_family_base}-Oblique"
            else:
                font_full_name = f"{f_family_base}-Italic"
        elif f_style == 'bold-italic':
            if f_family_base in ['Helvetica', 'Courier-New']:
                font_full_name = f"{f_family_base}-Bold-Oblique"
            else:
                font_full_name = f"{f_family_base}-Bold-Italic"
        else:
            font_full_name = f_family_base

        # Resolve text color
        text_color = caption_styles.get('textColor', 'white')
        if text_color.startswith('#'):
            # TextClip handles hex strings directly in some versions, but RGB is safer
            try:
                text_color = text_color # MoviePy TextClip usually handles hex
            except:
                text_color = 'white'
        else:
            text_color = color_map.get(text_color, 'white')

        # Resolve background color
        bg_style = caption_styles.get('background', 'semi-black')
        if bg_style.startswith('#'):
            # Convert hex to RGBA with 0.8 opacity for custom background
            rgb = hex_to_rgb(bg_style)
            bg_color = (rgb[0], rgb[1], rgb[2], 0.8)
        else:
            bg_color = bg_map.get(bg_style, (0, 0, 0, 0.8))
        
        # Resolve stroke color
        stroke_color = caption_styles.get('stockColor', 'black')
        if not stroke_color.startswith('#'):
            stroke_color = color_map.get(stroke_color, 'black')
        
        # Extract stroke width
        stroke_width = int(caption_styles.get('stockWidth', 2))
        
        progress_tracker[job_id] = {'step': 'Loading video', 'progress': 40}
        
        # Load video
        logger.info(f"Loading video from path: {video_path}")
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Close any existing video handles first
            import gc
            gc.collect()
            
            # Load video with error handling
            video = VideoFileClip(video_path)
            logger.info(f"Video loaded successfully. Duration: {video.duration}")
            
            # Ensure video has audio
            if video.audio is None:
                logger.warning("Video has no audio track")
                
        except Exception as e:
            logger.error(f"Error loading video: {str(e)}")
            raise
        
        progress_tracker[job_id] = {'step': 'Creating synchronized captions', 'progress': 60}
        
        # Create Whisper-synchronized caption clips
        logger.info(f"Caption text: '{caption_text}'")
        logger.info(f"Video duration: {video.duration}")
        
        # Create text clips for each word using Whisper timestamps
        caption_clips = []
        
        try:
            if not caption_timestamps:
                logger.warning("No caption timestamps available, using fallback timing")
                # Fallback to simple word timing if no timestamps
                words = caption_text.split()
                total_words = len(words)
                if total_words == 0:
                    final_video = video
                    return output_path
                    
                # Calculate timing to spread words across video duration
                available_duration = video.duration - 3
                time_per_word = available_duration / total_words
                
                # Calculate position based on style
                position = caption_styles.get('position', 'bottom')
                
                for i, word in enumerate(words):
                    # Clean word - remove newlines and extra spaces
                    clean_word = word.replace('\n', ' ').replace('\r', '').strip()
                    if not clean_word:
                        continue
                    
                    logger.info(f"Creating fallback caption for word: '{clean_word}'")
                    
                    # Calculate position based on style
                    if position == 'top':
                        y_pos = 100
                    elif position == 'middle':
                        y_pos = video.h // 2
                    else:  # bottom
                        y_pos = video.h - 100
                    
                    # Create caption text clip
                    # Clean word and ensure single line
                    clean_word = clean_word.replace('\n', ' ').strip()
                    
                    txt_clip = TextClip(
                        clean_word,
                        fontsize=f_size,
                        color=text_color,
                        font=font_full_name,
                        stroke_color=stroke_color if stroke_width > 0 else None,
                        stroke_width=stroke_width if stroke_width > 0 else 0,
                        method='label',
                        # Removed size and align=center to let 'label' naturally stay on one line
                    )
                    
                    # Add background if specified
                    if bg_color is not None:
                        from moviepy.editor import ColorClip
                        bg_width = txt_clip.w + 20
                        bg_height = txt_clip.h + 10
                        bg_clip = ColorClip(size=(bg_width, bg_height), color=bg_color[:3])
                        bg_clip = bg_clip.set_opacity(bg_color[3] if len(bg_color) > 3 else 0.8)
                        txt_clip = txt_clip.set_position('center')
                        txt_clip = CompositeVideoClip([bg_clip, txt_clip], size=(bg_width, bg_height))
                    
                    # Position caption based on style
                    txt_clip = txt_clip.set_position(('center', y_pos))
                    
                    # Calculate start time
                    start_time = i * time_per_word
                    word_duration = min(max(len(word) * 0.1 + 0.2, 0.3), time_per_word * 0.9)
                    
                    if start_time + word_duration > video.duration - 0.5:
                        word_duration = max(video.duration - 0.5 - start_time, 0.1)
                    
                    txt_clip = txt_clip.set_start(start_time).set_duration(word_duration)
                    caption_clips.append(txt_clip)
            else:
                # Group words into phrases for faster rendering (3x fewer clips)
                phrases = group_words_into_phrases(caption_timestamps, max_words_per_phrase=3)
                logger.info(f"Using Whisper timestamps: {len(caption_timestamps)} words grouped into {len(phrases)} phrases")
                
                # Use grouped phrases for rendering
                total_phrases = len(phrases)
                for i, phrase_data in enumerate(phrases):
                    phrase = phrase_data["text"]
                    start_time = phrase_data["start"]
                    duration = phrase_data["end"] - phrase_data["start"]
                    
                    # Clean phrase - remove newlines and extra spaces
                    clean_phrase = phrase.replace('\n', ' ').replace('\r', '').strip()
                    if not clean_phrase:
                        continue
                    
                    # Update live caption progress for UI
                    progress_tracker[job_id] = {
                        'step': f'Creating caption {i+1} of {total_phrases}',
                        'progress': 60 + int((i / total_phrases) * 20),  # 60-80% range
                        'status': 'rendering',
                        'caption_progress': {
                            'current': i + 1,
                            'total': total_phrases,
                            'text': clean_phrase[:30] + '...' if len(clean_phrase) > 30 else clean_phrase,
                            'time': start_time
                        }
                    }
                    
                    logger.info(f"Creating caption {i+1}: '{clean_phrase[:50]}...' at {start_time:.2f}s for {duration:.2f}s")
                    
                    # Calculate position based on style
                    position = caption_styles.get('position', 'bottom')
                    if position == 'top':
                        y_pos = 100
                    elif position == 'middle':
                        y_pos = video.h // 2
                    else:  # bottom
                        y_pos = video.h - 100
                    
                    # Create caption text clip
                    # Clean phrase and ensure single line
                    clean_phrase = clean_phrase.replace('\n', ' ').strip()
                    
                    txt_clip = TextClip(
                        clean_phrase,
                        fontsize=f_size,
                        color=text_color,
                        font=font_full_name,
                        stroke_color=stroke_color if stroke_width > 0 else None,
                        stroke_width=stroke_width if stroke_width > 0 else 0,
                        method='label'
                    )
                    
                    # Add background if specified
                    if bg_color is not None:
                        # Ensure background encompasses the label
                        from moviepy.editor import ColorClip
                        bg_width = txt_clip.w + 20
                        bg_height = txt_clip.h + 10
                        bg_clip = ColorClip(size=(bg_width, bg_height), color=bg_color[:3])
                        bg_clip = bg_clip.set_opacity(bg_color[3] if len(bg_color) > 3 else 0.8)
                        txt_clip = txt_clip.set_position('center')
                        txt_clip = CompositeVideoClip([bg_clip, txt_clip], size=(bg_width, bg_height))
                    
                    # Position caption based on style
                    txt_clip = txt_clip.set_position(('center', y_pos))
                    txt_clip = txt_clip.set_start(start_time).set_duration(duration)
                    
                    caption_clips.append(txt_clip)
            
            progress_tracker[job_id] = {'step': 'Rendering video with captions', 'progress': 80}
            
            # Composite video with all caption clips
            if caption_clips:
                logger.info(f"Compositing {len(caption_clips)} caption clips")
                final_video = CompositeVideoClip([video] + caption_clips)
            else:
                logger.warning("No caption clips to composite")
                final_video = video
            
        except Exception as e:
            logger.error(f"Error creating caption: {str(e)}")
            # Fallback to original video if caption creation fails
            final_video = video
        
        # Write output video with faster settings
        try:
            # Initialize our custom logger
            web_logger = MoviePyProgressLogger(job_id)
            
            # Use hardware acceleration if available, otherwise fallback to libx264
            # h264_nvenc is exponentially faster on NVIDIA GPUs
            try:
                final_video.write_videofile(
                    output_path, 
                    codec='h264_nvenc', 
                    audio_codec='aac', 
                    bitrate='4000k',
                    verbose=True, 
                    logger=web_logger,
                    threads=multiprocessing.cpu_count(),
                    preset='p1', 
                    fps=24, 
                    audio_fps=44100,
                    ffmpeg_params=['-pix_fmt', 'yuv420p'] # Ensure compatibility with standard players
                )
            except Exception as nvenc_e:
                logger.warning(f"NVENC failed, falling back to libx264: {str(nvenc_e)}")
                final_video.write_videofile(
                    output_path, 
                    codec='libx264', 
                    audio_codec='aac', 
                    bitrate='4000k',
                    verbose=False, 
                    logger=web_logger,
                    threads=multiprocessing.cpu_count(),
                    preset='ultrafast',
                    fps=24,
                    audio_fps=44100,
                    ffmpeg_params=['-pix_fmt', 'yuv420p'] # Ensure compatibility with standard players
                )
        except Exception as e:
            logger.error(f"Error writing video: {str(e)}")
            # Try to remove any existing file first
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
                    logger.info(f"Removed existing file: {output_path}")
                    # Retry writing
                    final_video.write_videofile(
                        output_path, 
                        codec='libx264', 
                        audio_codec='aac', 
                        verbose=False, 
                        logger=None,
                        threads=4,
                        preset='fast'
                    )
            except Exception as retry_e:
                logger.error(f"Retry also failed: {str(retry_e)}")
            raise
        
        progress_tracker[job_id] = {'step': 'Finalizing', 'progress': 100}
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error adding captions: {str(e)}")
        progress_tracker[job_id] = {'step': f'Error: {str(e)}', 'progress': -1}
        raise
    finally:
        # Properly close all clips to release file handles
        if video:
            try:
                video.close()
            except:
                pass
        if final_video:
            try:
                final_video.close()
            except:
                pass
        
        # Close individual word clips
        if 'word_clips' in locals():
            for clip in word_clips:
                try:
                    clip.close()
                except:
                    pass
        
        # Force garbage collection
        gc.collect()

def process_video_background(job_id, video_path, filename):
    """Background processing function"""
    try:
        progress_tracker[job_id] = {'step': 'Extracting audio', 'progress': 10}
        
        # Extract audio
        audio_path = extract_audio_from_video(video_path)
        
        progress_tracker[job_id] = {'step': 'Transcribing audio', 'progress': 30}
        
        # Transcribe audio with Whisper timestamps
        progress_tracker[job_id]['step'] = 'Transcribing with Whisper (this may take a moment...)'
        progress_tracker[job_id]['progress'] = 25
        
        caption_text, caption_timestamps = transcribe_audio_with_timestamps(audio_path)
        
        # Create preview
        preview_path = create_caption_preview(video_path, caption_text, job_id)
        
        # Wait for user to click render button
        progress_tracker[job_id] = {
            'step': 'Ready to render - Click Render button below',
            'progress': 40,
            'caption_text': caption_text,
            'caption_timestamps': caption_timestamps,
            'preview_filename': os.path.basename(preview_path) if preview_path and os.path.exists(preview_path) else None,
            'original_filename': filename,
            'video_path': video_path,
            'filename': filename,
            'status': 'ready_to_render'
        }
        save_progress()
        
        # Clean up temporary audio file
        if audio_path:
            safe_remove_file(audio_path)
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        progress_tracker[job_id] = {
            'step': f'Error: {str(e)}', 
            'progress': -1,
            'status': 'error',
            'error': str(e)
        }
        
        # Clean up files on error
        if 'audio_path' in locals() and audio_path:
            safe_remove_file(audio_path)
        safe_remove_file(video_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/progress/<job_id>')
def get_progress(job_id):
    """Get progress of video processing"""
    progress = progress_tracker.get(job_id, {})
    
    # Return error as JSON instead of redirecting
    if not progress:
        return jsonify({'error': 'Job not found', 'progress': -1, 'status': 'error'}), 404
    
    # If job is completed, return completed status without stuck data
    if progress.get('status') == 'completed':
        return jsonify({
            'step': 'Complete',
            'progress': 100,
            'status': 'completed',
            'output_filename': progress.get('output_filename'),
            'caption_text': progress.get('caption_text'),
            'original_filename': progress.get('original_filename'),
            'preview_filename': progress.get('preview_filename')
        })
    
    return jsonify(progress)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize progress tracking with video_path and filename for resume capability
        progress_tracker[job_id] = {
            'step': 'Starting...', 
            'progress': 0,
            'video_path': video_path,
            'filename': filename
        }
        save_progress()
        
        # For serverless, process synchronously or use async patterns
        # For local/dev, use threading
        if IS_SERVERLESS:
            # In serverless, we need to process immediately since threads don't persist
            # Note: This may timeout for large videos on Vercel's free tier (10s limit)
            try:
                process_video_background(job_id, video_path, filename)
            except Exception as e:
                logger.error(f"Serverless processing error: {e}")
                progress_tracker[job_id] = {
                    'step': f'Error: {str(e)}', 
                    'progress': -1,
                    'status': 'error'
                }
        else:
            # Start background processing for local development
            thread = threading.Thread(target=process_video_background, args=(job_id, video_path, filename))
            thread.daemon = True
            thread.start()
        
        # Return job ID for progress tracking
        return jsonify({'job_id': job_id, 'status': 'processing_started'})
    
    else:
        return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400

@app.route('/render/<job_id>', methods=['GET', 'POST'])
def render_video(job_id):
    """Start rendering the video with captions"""
    progress = progress_tracker.get(job_id, {})
    
    if progress.get('status') != 'ready_to_render':
        return jsonify({'error': 'Video not ready for rendering'}), 400
    
    try:
        # Get data from request (styles and potentially updated captions)
        request_data = request.get_json() or {}
        caption_styles = request_data.get('styles', request_data)
        updated_captions = request_data.get('captions')
        
        # Get video path and caption data
        video_path = progress.get('video_path')
        caption_text = updated_captions if updated_captions and isinstance(updated_captions, str) else progress.get('caption_text')
        caption_timestamps = updated_captions if updated_captions and isinstance(updated_captions, list) else progress.get('caption_timestamps', [])

        
        # Start rendering
        def render_background():
            try:
                progress_tracker[job_id] = {'step': 'Reusing transcription... Applying visual styles', 'progress': 50}
                save_progress()
                
                filename = progress.get('filename')
                
                logger.info(f"Starting render with video_path: {video_path}")
                logger.info(f"Filename: {filename}")
                logger.info(f"Caption text length: {len(caption_text) if caption_text else 0}")
                logger.info(f"Timestamps count: {len(caption_timestamps) if caption_timestamps else 0}")
                
                if not video_path or not filename:
                    raise ValueError("Missing video path or filename")
                
                # Add captions to video
                output_filename = f"captioned_{filename}"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                add_captions_to_video(video_path, caption_text, caption_timestamps, output_path, job_id, caption_styles)
                
                # Store final results with caption styles for re-rendering
                final_progress = {
                    'step': 'Complete', 
                    'progress': 100,
                    'output_filename': output_filename,
                    'caption_text': caption_text,
                    'caption_timestamps': caption_timestamps,
                    'original_filename': filename,
                    'preview_filename': progress.get('preview_filename'),
                    'status': 'completed',
                    'video_path': video_path,
                    'caption_styles': caption_styles
                }
                progress_tracker[job_id] = final_progress
                save_progress()
                logger.info(f"Render completed successfully for job {job_id}")
                
            except Exception as e:
                logger.error(f"Rendering error: {str(e)}")
                progress_tracker[job_id] = {
                    'step': f'Rendering error: {str(e)}', 
                    'progress': -1,
                    'status': 'error',
                    'error': str(e)
                }
        
        # For serverless, render synchronously; for local, use threading
        if IS_SERVERLESS:
            try:
                render_background()
                return jsonify({"status": "rendering_completed", "message": "Rendering completed"})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            # Start background thread for local development
            thread = threading.Thread(target=render_background)
            thread.daemon = True
            thread.start()
            return jsonify({"status": "rendering_started", "message": "Rendering started successfully"})
    
    except Exception as e:
        logger.error(f"Error starting render: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/result/<job_id>')
def show_result(job_id):
    """Show result page"""
    progress = progress_tracker.get(job_id, {})
    
    if progress.get('status') == 'completed':
        return render_template('result.html', 
                             original_filename=progress.get('original_filename'),
                             output_filename=progress.get('output_filename'),
                             caption_text=progress.get('caption_text'),
                             preview_filename=progress.get('preview_filename'))
    else:
        flash('Processing not completed or failed')
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serve files from uploads directory"""
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return "File not found", 404
    except Exception as e:
        logger.error(f"Error serving upload file: {str(e)}")
        return "Error serving file", 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            flash('File not found')
            return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        flash(f'Error downloading file: {str(e)}')
        return redirect(url_for('index'))

@app.route('/re-render/<job_id>', methods=['POST'])
def re_render_video(job_id):
    """Re-render video with new caption styles"""
    progress = progress_tracker.get(job_id, {})
    
    if not progress.get('video_path') or not progress.get('caption_text'):
        return jsonify({'error': 'Video data not available for re-rendering'}), 400
    
    try:
        # Get new data from request (styles and potentially updated captions)
        request_data = request.get_json() or {}
        new_styles = request_data.get('styles', request_data) # Support both flat and nested payload
        updated_captions = request_data.get('captions')
        
        # Get stored video data
        video_path = progress.get('video_path')
        caption_text = updated_captions if updated_captions and isinstance(updated_captions, str) else progress.get('caption_text')
        caption_timestamps = updated_captions if updated_captions and isinstance(updated_captions, list) else progress.get('caption_timestamps', [])

        filename = progress.get('filename') or progress.get('original_filename')
        
        # Start re-rendering in background
        def re_render_background():
            try:
                progress_tracker[job_id] = {
                    'step': 'Reusing transcription... Re-rendering with new styles',
                    'progress': 50,
                    'status': 're-rendering'
                }
                save_progress()
                
                logger.info(f"Starting re-render for job {job_id}")
                logger.info(f"New styles: {new_styles}")
                
                # Remove old output file if exists
                old_output = progress.get('output_filename')
                if old_output:
                    old_path = os.path.join(OUTPUT_FOLDER, old_output)
                    safe_remove_file(old_path)
                
                # Generate new output filename
                output_filename = f"captioned_{filename}"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                
                # Re-render with new styles
                add_captions_to_video(video_path, caption_text, caption_timestamps, output_path, job_id, new_styles)
                
                # Update progress with new results
                progress_tracker[job_id] = {
                    'step': 'Complete', 
                    'progress': 100,
                    'output_filename': output_filename,
                    'caption_text': caption_text,
                    'caption_timestamps': caption_timestamps,
                    'original_filename': filename,
                    'preview_filename': progress.get('preview_filename'),
                    'status': 'completed',
                    'video_path': video_path,
                    'caption_styles': new_styles
                }
                logger.info(f"Re-render completed successfully for job {job_id}")
                
            except Exception as e:
                logger.error(f"Re-rendering error: {str(e)}")
                progress_tracker[job_id] = {
                    'step': f'Re-rendering error: {str(e)}', 
                    'progress': -1,
                    'status': 'error',
                    'error': str(e)
                }
        
        # For serverless, re-render synchronously; for local, use threading
        if IS_SERVERLESS:
            try:
                re_render_background()
                return jsonify({"status": "re-rendering_completed", "message": "Re-rendering completed with new styles"})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        else:
            # Start background thread for local development
            thread = threading.Thread(target=re_render_background)
            thread.daemon = True
            thread.start()
            return jsonify({"status": "re-rendering_started", "message": "Re-rendering started with new styles"})
    
    except Exception as e:
        logger.error(f"Error starting re-render: {str(e)}")
        return jsonify({'error': str(e)}), 500

def safe_remove_file(file_path):
    """Safely remove a file with retry mechanism"""
    max_retries = 10
    for i in range(max_retries):
        try:
            if os.path.exists(file_path):
                # Try to close any potential file handles
                try:
                    # Force garbage collection to release file handles
                    gc.collect()
                    time.sleep(0.5)
                    os.remove(file_path)
                    logger.info(f"Successfully removed file: {file_path}")
                    return True
                except Exception as remove_error:
                    logger.warning(f"Remove attempt {i+1} failed: {str(remove_error)}")
                    if i < max_retries - 1:
                        time.sleep(2)  # Wait longer before retry
                        continue
                    else:
                        logger.error(f"Could not remove file after {max_retries} attempts: {file_path}")
                        return False
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {str(e)}")
            return False
    return False

# Load any saved progress on startup (local only)
load_progress()

# Vercel serverless entry point
app.debug = not IS_SERVERLESS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
