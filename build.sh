#!/bin/bash

# Build script for Vercel deployment
# Installs FFmpeg and ImageMagick for moviepy

echo "Starting build process..."

# Update package list
apt-get update -qq

# Install FFmpeg
apt-get install -y ffmpeg

# Install ImageMagick
apt-get install -y imagemagick

# Verify installations
echo "FFmpeg version:"
ffmpeg -version | head -1

echo "ImageMagick version:"
convert --version | head -1

echo "Build dependencies installed successfully!"
