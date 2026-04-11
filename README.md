# Video Caption Generator

A web application that automatically adds captions to videos using speech-to-text technology.

## Features

- 🎬 **Automatic Speech-to-Text**: Transcribes audio from video files using Google's speech recognition
- 📹 **Multiple Video Formats**: Supports MP4, AVI, MOV, MKV, WMV, FLV
- 🎨 **Embedded Captions**: Adds captions directly to the video with customizable styling
- 📥 **Easy Download**: Download processed videos with embedded captions
- 🌐 **Web Interface**: User-friendly drag-and-drop interface

## Requirements

- Python 3.7+
- FFmpeg (required by moviepy)

## Installation

1. **Clone or download the project files to your directory**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg:**
   
   **Windows:**
   - Download FFmpeg from https://ffmpeg.org/download.html
   - Extract and add the `bin` folder to your system PATH
   
   **Mac:**
   ```bash
   brew install ffmpeg
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

## Usage

1. **Start the application:**
   ```bash
   python app.py
   ```

2. **Open your web browser and go to:**
   ```
   http://localhost:5000
   ```

3. **Upload your video:**
   - Drag and drop a video file onto the upload area
   - Or click "Choose Video File" to browse
   - Click "Generate Captions"

4. **Download the processed video:**
   - Wait for processing to complete
   - View the generated captions
   - Click "Download Video" to save the captioned video

## How It Works

1. **Audio Extraction**: Extracts audio from the uploaded video file
2. **Speech Recognition**: Uses Google's speech recognition API to transcribe audio to text
3. **Caption Embedding**: Adds the transcribed text as captions at the bottom of the video
4. **Video Generation**: Creates a new video file with embedded captions

## File Structure

```
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/
│   ├── index.html      # Upload interface
│   └── result.html     # Results page
├── uploads/           # Temporary upload folder (auto-created)
├── output/            # Processed videos folder (auto-created)
└── README.md          # This file
```

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)
- WMV (.wmv)
- FLV (.flv)

## Technical Details

- **Backend**: Flask web framework
- **Speech Recognition**: SpeechRecognition library with Google API
- **Video Processing**: MoviePy for video editing and caption embedding
- **Frontend**: HTML5, CSS3, JavaScript with drag-and-drop functionality

## Limitations

- Requires internet connection for Google Speech Recognition API
- Processing time depends on video length and size
- Maximum file size: 500MB (configurable in app.py)
- Caption accuracy depends on audio quality and speech clarity

## Troubleshooting

**Common Issues:**

1. **"FFmpeg not found" error:**
   - Make sure FFmpeg is installed and added to system PATH

2. **"Speech recognition could not understand audio":**
   - Check if the video has clear speech
   - Ensure audio quality is good
   - Try with a shorter video clip

3. **"File too large" error:**
   - The default limit is 500MB, can be adjusted in `app.py`

4. **Processing takes too long:**
   - Processing time depends on video length
   - Larger videos take more time to process

## Security Notes

- Uploaded files are temporarily stored in the `uploads/` folder
- Processed files are stored in the `output/` folder
- Files are automatically cleaned up on errors
- Consider implementing file cleanup for production use

## License

This project is for educational and personal use.
