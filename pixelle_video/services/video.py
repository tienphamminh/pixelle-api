# Copyright (C) 2025 AIDC-AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Video Processing Service

High-performance video composition service built on ffmpeg-python.

Features:
- Video concatenation
- Audio/video merging
- Background music addition
- Image to video conversion

Note: Requires FFmpeg to be installed on the system.
"""

import math
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Literal, Optional

import ffmpeg
from loguru import logger

from pixelle_video.utils.os_util import (
    get_resource_path,
    list_resource_files,
    resource_exists
)


def check_ffmpeg() -> None:
    """
    Check if FFmpeg is installed on the system
    
    Raises:
        RuntimeError: If FFmpeg is not found
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "FFmpeg not found. Please install it:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu/Debian: apt-get install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )


class VideoService:
    """
    Video compositor for common video processing tasks

    Uses ffmpeg-python for high-performance video processing.
    All operations preserve video quality when possible (stream copy).

    Examples:
        >>> compositor = VideoCompositor()
        >>>
        >>> # Concatenate videos
        >>> compositor.concat_videos(
        ...     ["intro.mp4", "main.mp4", "outro.mp4"],
        ...     "final.mp4"
        ... )
        >>>
        >>> # Add voiceover
        >>> compositor.merge_audio_video(
        ...     "visual.mp4",
        ...     "voiceover.mp3",
        ...     "final.mp4"
        ... )
        >>>
        >>> # Add background music
        >>> compositor.add_bgm(
        ...     "video.mp4",
        ...     "music.mp3",
        ...     "final.mp4",
        ...     bgm_volume=0.3
        ... )
        >>>
        >>> # Create video from image + audio
        >>> compositor.create_video_from_image(
        ...     "frame.png",
        ...     "narration.mp3",
        ...     "segment.mp4"
        ... )
    """

    def __init__(self):
        self._ffmpeg_checked = False

    def _ensure_ffmpeg(self):
        """Lazily check FFmpeg availability on first use, not at import time"""
        if not self._ffmpeg_checked:
            check_ffmpeg()
            self._ffmpeg_checked = True

    def _probe_video_size(self, media_path: str) -> tuple[int, int]:
        """Return width and height for an image or video file."""
        probe = ffmpeg.probe(media_path)
        video_stream = next(
            (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"),
            None
        )
        if not video_stream:
            raise RuntimeError(f"No video stream found in media file: {media_path}")
        return int(video_stream["width"]), int(video_stream["height"])

    def _ken_burns_filter(
        self,
        image_stream,
        *,
        width: int,
        height: int,
        duration: float,
        fps: int,
        motion_strength: str,
        motion_preset: str,
        scene_index: int,
    ):
        """Apply a subtle Ken Burns camera move to a still image stream."""
        total_frames = max(1, int(math.ceil(duration * fps)))
        denominator = max(total_frames - 1, 1)
        # Mostly-linear easing: soft start/stop but tail velocity never collapses to ~0.
        # Pure cosine freezes the ends for ~200ms even after upscaling.
        progress_expr = (
            f"(0.7*(on/{denominator})+0.3*(0.5-0.5*cos(PI*on/{denominator})))"
        )

        strength_to_zoom = {
            "subtle": 1.08,
            "medium": 1.12,
            "strong": 1.18,
        }
        max_zoom = strength_to_zoom.get((motion_strength or "subtle").lower(), 1.08)

        auto_presets = (
            "slow_zoom_in",
            "slow_zoom_out",
            "pan_right",
            "pan_left",
            "pan_down",
            "pan_up",
            "drift_upper_right",
            "drift_lower_left",
            "zoom_in_left",
            "zoom_in_right",
        )
        valid_presets = {
            "slow_zoom_in",
            "slow_zoom_out",
            "pan_left",
            "pan_right",
            "pan_up",
            "pan_down",
            "drift_upper_left",
            "drift_upper_right",
            "drift_lower_left",
            "drift_lower_right",
            "push_in_left",
            "push_in_right",
            "push_out_left",
            "push_out_right",
            "zoom_in_left",
            "zoom_in_right",
        }
        requested_preset = (motion_preset or "auto").lower()
        preset = (
            auto_presets[scene_index % len(auto_presets)]
            if requested_preset == "auto" or requested_preset not in valid_presets
            else requested_preset
        )
        zoom_delta = max_zoom - 1.0

        if preset == "slow_zoom_out":
            z_expr = f"max({max_zoom:.5f}-{zoom_delta:.5f}*{progress_expr},1.0)"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"
        elif preset == "pan_right":
            z_expr = f"{max_zoom:.5f}"
            x_expr = f"(iw-iw/zoom)*{progress_expr}"
            y_expr = "ih/2-(ih/zoom/2)"
        elif preset == "pan_left":
            z_expr = f"{max_zoom:.5f}"
            x_expr = f"(iw-iw/zoom)*(1-{progress_expr})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif preset == "pan_up":
            z_expr = f"{max_zoom:.5f}"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = f"(ih-ih/zoom)*(1-{progress_expr})"
        elif preset == "pan_down":
            z_expr = f"{max_zoom:.5f}"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = f"(ih-ih/zoom)*{progress_expr}"
        elif preset == "drift_upper_left":
            z_expr = f"{max_zoom:.5f}"
            x_expr = f"(iw-iw/zoom)*(1-{progress_expr})"
            y_expr = f"(ih-ih/zoom)*(1-{progress_expr})"
        elif preset == "drift_upper_right":
            z_expr = f"{max_zoom:.5f}"
            x_expr = f"(iw-iw/zoom)*{progress_expr}"
            y_expr = f"(ih-ih/zoom)*(1-{progress_expr})"
        elif preset == "drift_lower_left":
            z_expr = f"{max_zoom:.5f}"
            x_expr = f"(iw-iw/zoom)*(1-{progress_expr})"
            y_expr = f"(ih-ih/zoom)*{progress_expr}"
        elif preset == "drift_lower_right":
            z_expr = f"{max_zoom:.5f}"
            x_expr = f"(iw-iw/zoom)*{progress_expr}"
            y_expr = f"(ih-ih/zoom)*{progress_expr}"
        elif preset == "push_in_left":
            z_expr = f"min(1.0+{zoom_delta:.5f}*{progress_expr},{max_zoom:.5f})"
            x_expr = f"(iw-iw/zoom)*(0.20+0.22*{progress_expr})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif preset == "push_in_right":
            z_expr = f"min(1.0+{zoom_delta:.5f}*{progress_expr},{max_zoom:.5f})"
            x_expr = f"(iw-iw/zoom)*(0.80-0.22*{progress_expr})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif preset == "push_out_left":
            z_expr = f"max({max_zoom:.5f}-{zoom_delta:.5f}*{progress_expr},1.0)"
            x_expr = f"(iw-iw/zoom)*(0.42-0.22*{progress_expr})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif preset == "push_out_right":
            z_expr = f"max({max_zoom:.5f}-{zoom_delta:.5f}*{progress_expr},1.0)"
            x_expr = f"(iw-iw/zoom)*(0.58+0.22*{progress_expr})"
            y_expr = "ih/2-(ih/zoom/2)"
        elif preset == "zoom_in_left":
            z_expr = f"min(1.0+{zoom_delta:.5f}*{progress_expr},{max_zoom:.5f})"
            x_expr = "(iw-iw/zoom)*0.25"
            y_expr = "ih/2-(ih/zoom/2)"
        elif preset == "zoom_in_right":
            z_expr = f"min(1.0+{zoom_delta:.5f}*{progress_expr},{max_zoom:.5f})"
            x_expr = "(iw-iw/zoom)*0.75"
            y_expr = "ih/2-(ih/zoom/2)"
        else:
            z_expr = f"min(1.0+{zoom_delta:.5f}*{progress_expr},{max_zoom:.5f})"
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "ih/2-(ih/zoom/2)"

        logger.debug(
            f"Applying Ken Burns preset={preset}, strength={motion_strength}, "
            f"zoom={max_zoom:.2f}, frames={total_frames}, fps={fps}"
        )

        # --- Ken Burns sub-pixel fix ---
        # zoompan rounds the crop origin (x,y)/zoom to INTEGER input pixels every frame
        # (no sub-pixel crop). With input pinned at output size (1080x1920) a subtle
        # ~0.15 px/frame pan holds the same pixel for several frames then snaps 1px ->
        # the visible staircase judder. Fix: upscale the still BEFORE zoompan and let
        # zoompan downscale back via s= so the crop origin lands on a much finer grid.
        # format=yuv444p is REQUIRED: otherwise the encoder's yuv420p propagates back and
        # zoompan floors x,y to EVEN input pixels (2px quantum -> double the judder).
        upscale_factor = 4  # speed-first default; raise to 6/8 if a scene still steps
        up_w = int(math.ceil(width * upscale_factor / 2) * 2)
        up_h = int(math.ceil(height * upscale_factor / 2) * 2)

        return (
            image_stream
            .filter("scale", width, height, force_original_aspect_ratio="increase")
            .filter("crop", width, height)
            .filter("scale", up_w, up_h, flags="lanczos")  # sub-pixel headroom
            .filter("format", "yuv444p")                   # 1px (not 2px) zoompan quantum
            .filter(
                "zoompan",
                z=z_expr,
                x=x_expr,
                y=y_expr,
                d=total_frames,
                s=f"{width}x{height}",   # zoompan downscales back to output size
                fps=fps,
            )
            .filter("setsar", 1)
        )

    def _format_ass_time(self, seconds: float) -> str:
        """Format seconds as ASS timestamp h:mm:ss.cc."""
        seconds = max(0.0, seconds)
        centiseconds = int(round(seconds * 100))
        cs = centiseconds % 100
        total_seconds = centiseconds // 100
        s = total_seconds % 60
        total_minutes = total_seconds // 60
        m = total_minutes % 60
        h = total_minutes // 60
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def _escape_ass_text(self, text: str) -> str:
        """Escape text so it is safe inside ASS dialogue lines."""
        return (
            str(text)
            .replace("\\", "")
            .replace("{", "(")
            .replace("}", ")")
            .replace("\n", " ")
            .strip()
        )

    def _subtitle_chunks(
        self,
        text: str,
        *,
        duration: float,
        words_per_chunk: int,
    ) -> list[tuple[float, float, str]]:
        """Split narration into timed phrase chunks without word-level alignment."""
        cleaned_text = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned_text:
            return []

        words = cleaned_text.split()
        target_words = min(max(int(words_per_chunk or 4), 2), 8)
        max_chars = 34
        break_words = {
            "but", "because", "while", "then", "so", "when", "before",
            "after", "until", "though", "although", "however",
        }

        chunks: list[str] = []
        current: list[str] = []
        for word in words:
            current.append(word)
            current_text = " ".join(current)
            clean_word = re.sub(r"[^a-z0-9']", "", word.lower())
            punctuation_break = word.rstrip().endswith((".", ",", ";", ":", "!", "?"))
            semantic_break = clean_word in break_words and len(current) >= 3
            length_break = len(current_text) >= max_chars
            count_break = len(current) >= target_words

            if (
                (len(current) >= 3 and (punctuation_break or semantic_break))
                or count_break
                or length_break
            ):
                chunks.append(current_text)
                current = []

        if current:
            chunks.append(" ".join(current))

        usable_duration = max(duration - 0.18, 0.25)
        max_chunks = max(1, int(usable_duration / 0.58))
        while len(chunks) > max_chunks:
            merged: list[str] = []
            i = 0
            merges_needed = len(chunks) - max_chunks
            merges_done = 0
            while i < len(chunks):
                if i + 1 < len(chunks) and merges_done < merges_needed:
                    merged.append(f"{chunks[i]} {chunks[i + 1]}")
                    i += 2
                    merges_done += 1
                else:
                    merged.append(chunks[i])
                    i += 1
            chunks = merged

        start_pad = min(0.10, duration * 0.05)
        end_pad = min(0.08, duration * 0.04)
        usable_duration = max(duration - start_pad - end_pad, 0.20)
        weights = [max(1, len(chunk.split())) for chunk in chunks]
        total_weight = sum(weights) or 1

        timed_chunks: list[tuple[float, float, str]] = []
        current_time = start_pad
        elapsed_weight = 0
        for index, (chunk, weight) in enumerate(zip(chunks, weights)):
            elapsed_weight += weight
            if index == len(chunks) - 1:
                end_time = max(current_time + 0.20, duration - end_pad)
            else:
                end_time = start_pad + usable_duration * elapsed_weight / total_weight
                end_time = max(end_time, current_time + 0.28)
            end_time = min(end_time, duration)
            timed_chunks.append((current_time, end_time, chunk))
            current_time = end_time

        return timed_chunks

    def _highlight_subtitle_word(self, words: list[str]) -> int:
        """Return the index of a word worth highlighting in a phrase chunk."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "to", "of", "in", "on",
            "for", "with", "without", "into", "from", "his", "her", "their",
            "he", "she", "it", "they", "is", "are", "was", "were", "be",
            "been", "being", "this", "that", "these", "those", "who", "what",
            "when", "where", "why", "how", "not", "can", "will", "would",
        }
        best_index = -1
        best_score = -1
        for index, word in enumerate(words):
            clean_word = re.sub(r"[^A-Za-z0-9]", "", word).lower()
            if len(clean_word) < 4 or clean_word in stopwords:
                continue
            score = len(clean_word)
            if score > best_score:
                best_index = index
                best_score = score
        return best_index

    def _format_subtitle_dialogue_text(
        self,
        chunk: str,
        *,
        style: str,
        max_line_chars: int,
    ) -> str:
        """Format one subtitle chunk with optional line break and keyword color."""
        words = chunk.split()
        if not words:
            return ""

        clean_style = (style or "comic_tiktok").lower()
        highlight_index = -1
        if clean_style == "yellow_keyword":
            highlight_index = self._highlight_subtitle_word(words)

        break_after = -1
        if len(chunk) > max_line_chars and len(words) > 2:
            cumulative = 0
            midpoint = len(chunk) / 2
            best_distance = len(chunk)
            for index in range(1, len(words)):
                cumulative += len(words[index - 1]) + 1
                distance = abs(cumulative - midpoint)
                if distance < best_distance:
                    best_distance = distance
                    break_after = index - 1

        formatted_words: list[str] = []
        for index, word in enumerate(words):
            escaped_word = self._escape_ass_text(word)
            if index == highlight_index:
                escaped_word = r"{\c&H0000D7FF&}" + escaped_word + r"{\c&H00FFFFFF&}"
            formatted_words.append(escaped_word)
            if index == break_after:
                formatted_words.append(r"\N")

        text = " ".join(formatted_words).replace(r"\N ", r"\N")
        if clean_style == "comic_tiktok":
            return r"{\an2\fscx96\fscy96\t(0,120,\fscx108\fscy108)\t(120,260,\fscx100\fscy100)}" + text
        return r"{\an2}" + text

    def create_phrase_subtitle_file(
        self,
        text: str,
        output: str,
        *,
        duration: float,
        width: int,
        height: int,
        style: str = "comic_tiktok",
        words_per_chunk: int = 4,
    ) -> str:
        """Create an ASS subtitle file with timed phrase chunks."""
        timed_chunks = self._subtitle_chunks(
            text,
            duration=duration,
            words_per_chunk=words_per_chunk,
        )

        clean_style = (style or "comic_tiktok").lower()
        font_size_ratio = 0.067 if clean_style == "comic_tiktok" else 0.058
        outline_ratio = 0.008 if clean_style == "comic_tiktok" else 0.006
        shadow_ratio = 0.003 if clean_style == "comic_tiktok" else 0.002
        font_size = max(28, int(width * font_size_ratio))
        outline = max(3, int(width * outline_ratio))
        shadow = max(1, int(width * shadow_ratio))
        margin_v = max(48, int(height * 0.118))
        max_line_chars = 22 if clean_style == "comic_tiktok" else 28

        if clean_style == "yellow_keyword":
            primary_color = "&H00FFFFFF"
            outline_color = "&H00000000"
        elif clean_style == "clean_white":
            primary_color = "&H00FFFFFF"
            outline_color = "&H00000000"
        else:
            primary_color = "&H00FFFFFF"
            outline_color = "&H00000000"

        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "",
            "[V4+ Styles]",
            (
                "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
                "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
                "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
                "MarginR,MarginV,Encoding"
            ),
            (
                f"Style: Default,Arial,{font_size},{primary_color},&H0000D7FF,"
                f"{outline_color},&H8C000000,-1,0,0,0,100,100,0,0,1,"
                f"{outline},{shadow},2,64,64,{margin_v},1"
            ),
            "",
            "[Events]",
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
        ]

        for start, end, chunk in timed_chunks:
            dialogue_text = self._format_subtitle_dialogue_text(
                chunk,
                style=clean_style,
                max_line_chars=max_line_chars,
            )
            if dialogue_text:
                lines.append(
                    f"Dialogue: 0,{self._format_ass_time(start)},"
                    f"{self._format_ass_time(end)},Default,,0,0,0,,{dialogue_text}"
                )

        os.makedirs(os.path.dirname(output), exist_ok=True)
        Path(output).write_text("\n".join(lines), encoding="utf-8")
        logger.debug(f"Phrase subtitle file created: {output} ({len(timed_chunks)} chunks)")
        return output

    def _escape_subtitle_filter_path(self, subtitle_path: str) -> str:
        """Escape a local path for FFmpeg's subtitles filter."""
        resolved_path = Path(subtitle_path).resolve()
        try:
            path = resolved_path.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            path = resolved_path.as_posix().replace(":", r"\:")
        return (
            path
            .replace("\\", "/")
            .replace("'", r"\'")
            .replace(",", r"\,")
        )

    def burn_subtitles(
        self,
        video: str,
        subtitle_path: str,
        output: str,
    ) -> str:
        """Burn an ASS subtitle file into a video."""
        self._ensure_ffmpeg()
        escaped_subtitle_path = self._escape_subtitle_filter_path(subtitle_path)
        input_video = ffmpeg.input(video)
        video_stream = input_video.video.filter("subtitles", escaped_subtitle_path)
        output_kwargs = {
            "vcodec": "libx264",
            "acodec": "aac",
            "audio_bitrate": "192k",
            "pix_fmt": "yuv420p",
            "preset": "medium",
            "crf": 23,
        }

        try:
            if self.has_audio_stream(video):
                (
                    ffmpeg
                    .output(video_stream, input_video.audio, output, **output_kwargs)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            else:
                (
                    ffmpeg
                    .output(video_stream, output, vcodec="libx264", pix_fmt="yuv420p", preset="medium", crf=23)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            logger.success(f"Subtitles burned into video: {output}")
            return output
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg subtitle burn error: {error_msg}")
            raise RuntimeError(f"Failed to burn subtitles: {error_msg}")

    def concat_videos(
        self,
        videos: List[str],
        output: str,
        method: Literal["demuxer", "filter"] = "demuxer",
        target_fps: Optional[int] = None,
        bgm_path: Optional[str] = None,
        bgm_volume: float = 0.2,
        bgm_mode: Literal["once", "loop"] = "loop"
    ) -> str:
        """
        Concatenate multiple videos into one

        Args:
            videos: List of video file paths to concatenate
            output: Output video file path
            method: Concatenation method
                - "demuxer": Fast, no re-encoding (requires identical formats)
                - "filter": Slower but handles different formats
            bgm_path: Background music file path (optional)
                - None: No BGM
        """
        self._ensure_ffmpeg()

        if not videos:
            raise ValueError("Videos list cannot be empty")
        
        if len(videos) == 1:
            logger.info(f"Only one video provided, copying to {output}")
            shutil.copy(videos[0], output)
            return output
        
        logger.info(f"Concatenating {len(videos)} videos using {method} method")
        
        # Step 1: Concatenate videos
        if bgm_path:
            # If BGM needed, concatenate to temp file first
            temp_output = output.replace('.mp4', '_no_bgm.mp4')
            concat_result = self._concat_demuxer(videos, temp_output) if method == "demuxer" else self._concat_filter(videos, temp_output, target_fps)
            
            # Step 2: Add BGM
            logger.info(f"Adding BGM: {bgm_path} (volume={bgm_volume}, mode={bgm_mode})")
            final_result = self._add_bgm_to_video(
                video=concat_result,
                bgm_path=bgm_path,
                output=output,
                volume=bgm_volume,
                mode=bgm_mode
            )
            
            # Clean up temp file
            if os.path.exists(temp_output):
                os.unlink(temp_output)
            
            return final_result
        else:
            # No BGM, direct concatenation
            if method == "demuxer":
                return self._concat_demuxer(videos, output)
            else:
                return self._concat_filter(videos, output, target_fps)
    
    def _concat_demuxer(self, videos: List[str], output: str) -> str:
        """
        Concatenate using concat demuxer (fast, no re-encoding)
        
        FFmpeg equivalent:
            ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4
        """
        # Create temporary file list
        with tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            suffix='.txt',
            encoding='utf-8'
        ) as f:
            for video in videos:
                abs_path = Path(video).absolute()
                escaped_path = str(abs_path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
            filelist = f.name
        
        try:
            logger.debug(f"Created filelist: {filelist}")
            (
                ffmpeg
                .input(filelist, format='concat', safe=0)
                .output(output, c='copy')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.success(f"Videos concatenated successfully: {output}")
            return output
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg concat error: {error_msg}")
            raise RuntimeError(f"Failed to concatenate videos: {error_msg}")
        finally:
            if os.path.exists(filelist):
                os.unlink(filelist)
    
    def _concat_filter(self, videos: List[str], output: str, target_fps: Optional[int] = None) -> str:
        """
        Concatenate using concat filter (slower but handles different formats)
        
        FFmpeg equivalent:
            ffmpeg -i v1.mp4 -i v2.mp4 -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]"
                   -map "[v]" -map "[a]" output.mp4
        """
        try:
            # Build filter_complex string manually
            n = len(videos)
            
            # Build input stream labels: [0:v][0:a][1:v][1:a]...
            stream_spec = "".join([f"[{i}:v][{i}:a]" for i in range(n)])
            filter_complex = f"{stream_spec}concat=n={n}:v=1:a=1[v][a]"
            
            # Build ffmpeg command
            cmd = ['ffmpeg']
            for video in videos:
                cmd.extend(['-i', video])
            cmd.extend(['-filter_complex', filter_complex, '-map', '[v]', '-map', '[a]'])
            if target_fps:
                # Re-encode onto one uniform CFR timeline so scene joins don't freeze the
                # last frame (concat demuxer copy froze each seam by the segment's
                # audio>video surplus). Filter concat already re-encodes; pin fps here.
                cmd.extend(['-r', str(target_fps), '-vsync', 'cfr'])
            cmd.extend([
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-b:a', '192k',
                '-y',  # Overwrite output
                output,
            ])
            
            # Run command
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.success(f"Videos concatenated successfully: {output}")
            return output
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"FFmpeg concat filter error: {error_msg}")
            raise RuntimeError(f"Failed to concatenate videos: {error_msg}")
        except Exception as e:
            logger.error(f"Concatenation error: {e}")
            raise RuntimeError(f"Failed to concatenate videos: {e}")
    
    def _get_video_duration(self, video: str) -> float:
        """Get video duration in seconds"""
        try:
            probe = ffmpeg.probe(video)
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            logger.warning(f"Failed to get video duration: {e}")
            return 0.0
    
    def _get_audio_duration(self, audio: str) -> float:
        """Get audio duration in seconds"""
        try:
            probe = ffmpeg.probe(audio)
            duration = float(probe['format']['duration'])
            return duration
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}, using estimate")
            # Fallback: estimate based on file size (very rough)
            import os
            file_size = os.path.getsize(audio)
            # Assume ~16kbps for MP3, so 2KB per second
            estimated_duration = file_size / 2000
            return max(1.0, estimated_duration)  # At least 1 second
    
    def has_audio_stream(self, video: str) -> bool:
        """
        Check if video has audio stream
        
        Args:
            video: Video file path
        
        Returns:
            True if video has audio stream, False otherwise
        """
        try:
            probe = ffmpeg.probe(video)
            audio_streams = [s for s in probe.get('streams', []) if s['codec_type'] == 'audio']
            has_audio = len(audio_streams) > 0
            logger.debug(f"Video {video} has_audio={has_audio}")
            return has_audio
        except Exception as e:
            logger.warning(f"Failed to probe video audio streams: {e}, assuming no audio")
            return False
    
    def merge_audio_video(
        self,
        video: str,
        audio: str,
        output: str,
        replace_audio: bool = True,
        audio_volume: float = 1.0,
        video_volume: float = 0.0,
        pad_strategy: str = "freeze",  # "freeze" (freeze last frame) or "black" (black screen)
        auto_adjust_duration: bool = True,  # Automatically adjust video duration to match audio
        duration_tolerance: float = 0.3,  # Tolerance for video being longer than audio (seconds)
    ) -> str:
        """
        Merge audio with video with intelligent duration adjustment
        
        Automatically handles duration mismatches between video and audio:
        - If video < audio: Pad video to match audio (avoid black screen)
        - If video > audio (within tolerance): Keep as-is (acceptable)
        - If video > audio (exceeds tolerance): Trim video to match audio
        
        Automatically handles videos with or without audio streams.
        - If video has no audio: adds the audio track
        - If video has audio and replace_audio=True: replaces with new audio
        - If video has audio and replace_audio=False: mixes both audio tracks
        
        Args:
            video: Video file path
            audio: Audio file path
            output: Output video file path
            replace_audio: If True, replace video's audio; if False, mix with original
            audio_volume: Volume of the new audio (0.0 to 1.0+)
            video_volume: Volume of original video audio (0.0 to 1.0+)
                         Only used when replace_audio=False
            pad_strategy: Strategy to pad video if audio is longer
                         - "freeze": Freeze last frame (default)
                         - "black": Fill with black screen
            auto_adjust_duration: Enable intelligent duration adjustment (default: True)
            duration_tolerance: Tolerance for video being longer than audio in seconds (default: 0.3)
                              Videos within this tolerance won't be trimmed
        
        Returns:
            Path to the output video file
        
        Raises:
            RuntimeError: If FFmpeg execution fails
        
        Note:
            - Uses the longer duration between video and audio
            - When audio is longer, video is padded using pad_strategy
            - When video is longer, audio is looped or extended
            - Automatically detects if video has audio
            - When video is silent, audio is added regardless of replace_audio
            - When replace_audio=True and video has audio, original audio is removed
            - When replace_audio=False and video has audio, original and new audio are mixed
        """
        self._ensure_ffmpeg()

        # Get durations of video and audio
        video_duration = self._get_video_duration(video)
        audio_duration = self._get_audio_duration(audio)
        
        logger.info(f"Video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
        
        # Intelligent duration adjustment (if enabled)
        if auto_adjust_duration:
            diff = video_duration - audio_duration
            
            if diff < 0:
                # Video shorter than audio → Must pad to avoid black screen
                logger.warning(f"⚠️ Video shorter than audio by {abs(diff):.2f}s, padding required")
                video = self._pad_video_to_duration(video, audio_duration, pad_strategy)
                video_duration = audio_duration  # Update duration after padding
                logger.info(f"📌 Padded video to {audio_duration:.2f}s")
            
            elif diff > duration_tolerance:
                # Video significantly longer than audio → Trim
                logger.info(f"⚠️ Video longer than audio by {diff:.2f}s (tolerance: {duration_tolerance}s)")
                video = self._trim_video_to_duration(video, audio_duration)
                video_duration = audio_duration  # Update duration after trimming
                logger.info(f"✂️ Trimmed video to {audio_duration:.2f}s")
            
            else:  # 0 <= diff <= duration_tolerance
                # Video slightly longer but within tolerance → Keep as-is
                logger.info(f"✅ Duration acceptable: video={video_duration:.2f}s, audio={audio_duration:.2f}s (diff={diff:.2f}s)")
        
        # Determine target duration (max of both)
        target_duration = max(video_duration, audio_duration)
        logger.info(f"Target output duration: {target_duration:.2f}s")
        
        # Check if video has audio stream
        video_has_audio = self.has_audio_stream(video)
        
        # Prepare video stream (potentially with padding)
        input_video = ffmpeg.input(video)
        video_stream = input_video.video
        
        # Pad video if audio is longer
        if audio_duration > video_duration:
            pad_duration = audio_duration - video_duration
            logger.info(f"Audio is longer, padding video by {pad_duration:.2f}s using '{pad_strategy}' strategy")
            
            if pad_strategy == "freeze":
                # Freeze last frame: tpad filter
                video_stream = video_stream.filter('tpad', stop_mode='clone', stop_duration=pad_duration)
            else:  # black
                # Generate black frames for padding duration
                # Get video properties
                probe = ffmpeg.probe(video)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                width = int(video_info['width'])
                height = int(video_info['height'])
                fps_str = video_info['r_frame_rate']
                fps_num, fps_den = map(int, fps_str.split('/'))
                fps = fps_num / fps_den if fps_den != 0 else 30
                
                # Create black video for padding
                black_video_path = self._get_unique_temp_path("black_pad", os.path.basename(output))
                black_input = ffmpeg.input(
                    f'color=c=black:s={width}x{height}:r={fps}',
                    f='lavfi',
                    t=pad_duration
                )
                
                # Concatenate original video with black padding
                video_stream = ffmpeg.concat(video_stream, black_input.video, v=1, a=0)
        
        # Prepare audio stream (pad if needed to match target duration)
        input_audio = ffmpeg.input(audio)
        audio_stream = input_audio.audio.filter('volume', audio_volume)
        
        # Pad audio with silence if video is longer
        if video_duration > audio_duration:
            pad_duration = video_duration - audio_duration
            logger.info(f"Video is longer, padding audio with {pad_duration:.2f}s silence")
            # Use apad to add silence at the end
            audio_stream = audio_stream.filter('apad', whole_dur=target_duration)
        
        if not video_has_audio:
            logger.info(f"Video has no audio stream, adding audio track")
            # Video is silent, just add the audio
            try:
                (
                    ffmpeg
                    .output(
                        video_stream,
                        audio_stream,
                        output,
                        vcodec='libx264',  # Re-encode video if padded
                        acodec='aac',
                        audio_bitrate='192k'
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
                
                logger.success(f"Audio added to silent video: {output}")
                return output
            except ffmpeg.Error as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                logger.error(f"FFmpeg error adding audio to silent video: {error_msg}")
                raise RuntimeError(f"Failed to add audio to video: {error_msg}")
        
        # Video has audio, proceed with merging
        logger.info(f"Merging audio with video (replace={replace_audio})")
        
        try:
            if replace_audio:
                # Replace audio: use only new audio, ignore original
                (
                    ffmpeg
                    .output(
                        video_stream,
                        audio_stream,
                        output,
                        vcodec='libx264',  # Re-encode video if padded
                        acodec='aac',
                        audio_bitrate='192k'
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            else:
                # Mix audio: combine original and new audio
                mixed_audio = ffmpeg.filter(
                    [
                        input_video.audio.filter('volume', video_volume),
                        audio_stream
                    ],
                    'amix',
                    inputs=2,
                    duration='longest'  # Use longest audio
                )
                
                (
                    ffmpeg
                    .output(
                        video_stream,
                        mixed_audio,
                        output,
                        vcodec='libx264',  # Re-encode video if padded
                        acodec='aac',
                        audio_bitrate='192k'
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                )
            
            logger.success(f"Audio merged successfully: {output}")
            return output
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg merge error: {error_msg}")
            raise RuntimeError(f"Failed to merge audio and video: {error_msg}")
    
    def overlay_image_on_video(
        self,
        video: str,
        overlay_image: str,
        output: str,
        scale_mode: str = "contain"
    ) -> str:
        """
        Overlay a transparent image on top of video
        
        Args:
            video: Base video file path
            overlay_image: Transparent overlay image path (e.g., rendered HTML with transparent background)
            output: Output video file path
            scale_mode: How to scale the base video to fit the overlay size
                - "contain": Scale video to fit within overlay dimensions (letterbox/pillarbox)
                - "cover": Scale video to cover overlay dimensions (may crop)
                - "stretch": Stretch video to exact overlay dimensions
        
        Returns:
            Path to the output video file
        
        Raises:
            RuntimeError: If FFmpeg execution fails
        
        Note:
            - Overlay image should have transparent background
            - Video is scaled to match overlay dimensions based on scale_mode
            - Final video size matches overlay image size
            - Video codec is re-encoded to support overlay
        """
        self._ensure_ffmpeg()
        logger.info(f"Overlaying image on video (scale_mode={scale_mode})")
        
        try:
            # Get overlay image dimensions
            overlay_probe = ffmpeg.probe(overlay_image)
            overlay_stream = next(s for s in overlay_probe['streams'] if s['codec_type'] == 'video')
            overlay_width = int(overlay_stream['width'])
            overlay_height = int(overlay_stream['height'])
            
            logger.debug(f"Overlay dimensions: {overlay_width}x{overlay_height}")
            
            input_video = ffmpeg.input(video)
            input_overlay = ffmpeg.input(overlay_image)
            
            # Scale video to fit overlay size using scale_mode
            if scale_mode == "contain":
                # Scale to fit (letterbox/pillarbox if aspect ratio differs)
                # Use scale filter with force_original_aspect_ratio=decrease and pad to center
                scaled_video = (
                    input_video
                    .filter('scale', overlay_width, overlay_height, force_original_aspect_ratio='decrease')
                    .filter('pad', overlay_width, overlay_height, '(ow-iw)/2', '(oh-ih)/2', color='black')
                )
            elif scale_mode == "cover":
                # Scale to cover (crop if aspect ratio differs)
                scaled_video = (
                    input_video
                    .filter('scale', overlay_width, overlay_height, force_original_aspect_ratio='increase')
                    .filter('crop', overlay_width, overlay_height)
                )
            else:  # stretch
                # Stretch to exact dimensions
                scaled_video = input_video.filter('scale', overlay_width, overlay_height)
            
            # Overlay the transparent image on top of the scaled video
            output_stream = ffmpeg.overlay(scaled_video, input_overlay)
            
            (
                ffmpeg
                .output(output_stream, output, 
                        vcodec='libx264',
                        pix_fmt='yuv420p',
                        preset='medium',
                        crf=23)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.success(f"Image overlaid on video: {output}")
            return output
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg overlay error: {error_msg}")
            raise RuntimeError(f"Failed to overlay image on video: {error_msg}")
    
    def create_video_from_image(
        self,
        image: str,
        audio: str,
        output: str,
        fps: int = 30,
        motion: str = "static",
        motion_strength: str = "subtle",
        motion_preset: str = "auto",
        scene_index: int = 0,
    ) -> str:
        """
        Create video from static image and audio
        
        Args:
            image: Image file path
            audio: Audio file path
            output: Output video path
            fps: Frames per second
            motion: Image motion mode ("static" or "ken_burns")
            motion_strength: Strength for Ken Burns motion ("subtle", "medium", "strong")
            motion_preset: Ken Burns preset ("auto" rotates presets by scene)
            scene_index: Scene index used to rotate Ken Burns presets
        
        Returns:
            Path to the output video
        
        Raises:
            RuntimeError: If FFmpeg execution fails
        
        Note:
            - Image is displayed as static frame for the duration of audio by default
            - With motion="ken_burns", a subtle zoom/pan is applied to the still image
            - Video duration matches audio duration
            - Useful for creating video segments from storyboard frames
        
        Example:
            >>> compositor.create_video_from_image(
            ...     "frame.png",
            ...     "narration.mp3",
            ...     "segment.mp4"
            ... )
        """
        self._ensure_ffmpeg()
        motion = (motion or "static").lower()
        render_fps = max(fps, 60) if motion == "ken_burns" else fps
        logger.info(f"Creating video from image and audio (motion={motion}, fps={render_fps})")
        
        try:
            # Get audio duration to ensure exact video duration match
            probe = ffmpeg.probe(audio)
            audio_duration = float(probe['format']['duration'])
            logger.debug(f"Audio duration: {audio_duration:.3f}s")
            
            input_audio = ffmpeg.input(audio)

            if motion == "ken_burns":
                input_image = ffmpeg.input(image)
                width, height = self._probe_video_size(image)
                video_input = self._ken_burns_filter(
                    input_image.video,
                    width=width,
                    height=height,
                    duration=audio_duration,
                    fps=render_fps,
                    motion_strength=motion_strength,
                    motion_preset=motion_preset,
                    scene_index=scene_index,
                )
            else:
                # Input image with loop (loop=1 means loop indefinitely)
                input_image = ffmpeg.input(image, loop=1, framerate=render_fps)
                video_input = input_image.video
            
            # Combine image and audio
            # Use -t to explicitly set video duration = audio duration
            (
                ffmpeg
                .output(
                    video_input,
                    input_audio,
                    output,
                    t=audio_duration,  # Force video duration to match audio exactly
                    vcodec='libx264',
                    acodec='aac',
                    pix_fmt='yuv420p',
                    audio_bitrate='192k',
                    r=render_fps,
                    vsync='cfr',
                    preset='fast',  # slow zoom of a still compresses trivially; offsets upscale cost
                    crf=23,
                    **{'b:v': '2M'}  # Video bitrate
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.success(f"Video created from image: {output} (duration: {audio_duration:.3f}s)")
            return output
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error creating video from image: {error_msg}")
            raise RuntimeError(f"Failed to create video from image: {error_msg}")
    
    def add_bgm(
        self,
        video: str,
        bgm: str,
        output: str,
        bgm_volume: float = 0.3,
        loop: bool = True,
        fade_in: float = 0.0,
        fade_out: float = 0.0,
    ) -> str:
        """
        Add background music to video
        
        Args:
            video: Video file path
            bgm: Background music file path
            output: Output video file path
            bgm_volume: BGM volume relative to original (0.0 to 1.0+)
            loop: If True, loop BGM to match video duration
            fade_in: BGM fade-in duration in seconds
            fade_out: BGM fade-out duration in seconds (not yet implemented)
        
        Returns:
            Path to the output video file
        
        Raises:
            RuntimeError: If FFmpeg execution fails
        
        Note:
            - BGM is mixed with original video audio
            - If loop=True, BGM repeats until video ends
            - Fade effects are applied to BGM only
        """
        self._ensure_ffmpeg()
        logger.info(f"Adding BGM to video (volume={bgm_volume}, loop={loop})")
        
        try:
            input_video = ffmpeg.input(video)
            
            # Configure BGM input with looping if needed
            bgm_input = ffmpeg.input(
                bgm,
                stream_loop=-1 if loop else 0  # -1 = infinite loop
            )
            
            # Apply volume adjustment to BGM
            bgm_audio = bgm_input.audio.filter('volume', bgm_volume)
            
            # Apply fade effects if specified
            if fade_in > 0:
                bgm_audio = bgm_audio.filter('afade', type='in', duration=fade_in)
            # Note: fade_out at the end requires knowing the duration, which is complex
            # For now, we skip fade_out in this implementation
            # A more advanced implementation would need to:
            # 1. Get video duration
            # 2. Calculate fade_out start time
            # 3. Apply fade filter with specific start_time
            
            # Mix original audio with BGM
            mixed_audio = ffmpeg.filter(
                [input_video.audio, bgm_audio],
                'amix',
                inputs=2,
                duration='first'  # Use video's duration
            )
            
            (
                ffmpeg
                .output(
                    input_video.video,
                    mixed_audio,
                    output,
                    vcodec='copy',
                    acodec='aac',
                    audio_bitrate='192k'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            logger.success(f"BGM added successfully: {output}")
            return output
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg BGM error: {error_msg}")
            raise RuntimeError(f"Failed to add BGM: {error_msg}")
    
    def _add_bgm_to_video(
        self,
        video: str,
        bgm_path: str,
        output: str,
        volume: float = 0.2,
        mode: Literal["once", "loop"] = "loop"
    ) -> str:
        """
        Internal helper to add BGM to video with path resolution
        
        Args:
            video: Video file path
            bgm_path: BGM path (can be preset name or custom path)
            output: Output file path
            volume: BGM volume (0.0-1.0)
            mode: "once" or "loop"
        
        Returns:
            Path to output video
        
        Raises:
            FileNotFoundError: If BGM file not found
        """
        # Resolve BGM path (raises FileNotFoundError if not found)
        resolved_bgm = self._resolve_bgm_path(bgm_path)
        
        # Add BGM using existing method
        loop = (mode == "loop")
        return self.add_bgm(
            video=video,
            bgm=resolved_bgm,
            output=output,
            bgm_volume=volume,
            loop=loop,
            fade_in=0.0
        )
    
    def _get_unique_temp_path(self, prefix: str, original_filename: str) -> str:
        """
        Generate unique temporary file path to avoid concurrent conflicts
        
        Args:
            prefix: Prefix for the temp file (e.g., "trimmed", "padded", "black_pad")
            original_filename: Original filename to preserve in temp path
        
        Returns:
            Unique temporary file path with format: temp/{prefix}_{uuid}_{original_filename}
        
        Example:
            >>> self._get_unique_temp_path("trimmed", "video.mp4")
            >>> # Returns: "temp/trimmed_a3f2d8c1_video.mp4"
        """
        from pixelle_video.utils.os_util import get_temp_path
        
        unique_id = uuid.uuid4().hex[:8]
        return get_temp_path(f"{prefix}_{unique_id}_{original_filename}")
    
    def _resolve_bgm_path(self, bgm_path: str) -> str:
        """
        Resolve BGM path (filename or custom path) with custom override support
        
        Search priority:
            1. Direct path (absolute or relative)
            2. data/bgm/{filename} (custom)
            3. bgm/{filename} (default)
        
        Args:
            bgm_path: Can be:
                - Filename with extension (e.g., "default.mp3", "happy.mp3"): auto-resolved from bgm/ or data/bgm/
                - Custom file path (absolute or relative)
        
        Returns:
            Resolved absolute path
        
        Raises:
            FileNotFoundError: If BGM file not found
        """
        # Try direct path first (absolute or relative)
        if os.path.exists(bgm_path):
            return os.path.abspath(bgm_path)
        
        # Try as filename in resource directories (custom > default)
        if resource_exists("bgm", bgm_path):
            return get_resource_path("bgm", bgm_path)
        
        # Not found - provide helpful error message
        tried_paths = [
            os.path.abspath(bgm_path),
            f"data/bgm/{bgm_path} or bgm/{bgm_path}"
        ]
        
        # List available BGM files
        available_bgm = self._list_available_bgm()
        available_msg = f"\n  Available BGM files: {', '.join(available_bgm)}" if available_bgm else ""
        
        raise FileNotFoundError(
            f"BGM file not found: '{bgm_path}'\n"
            f"  Tried paths:\n"
            f"    1. {tried_paths[0]}\n"
            f"    2. {tried_paths[1]}"
            f"{available_msg}"
        )
    
    def _list_available_bgm(self) -> list[str]:
        """
        List available BGM files (merged from bgm/ and data/bgm/)
        
        Returns:
            List of filenames (with extensions), sorted
        """
        try:
            # Use resource API to get merged list
            all_files = list_resource_files("bgm")
            
            # Filter to audio files only
            audio_extensions = ('.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac')
            return sorted([f for f in all_files if f.lower().endswith(audio_extensions)])
        except Exception as e:
            logger.warning(f"Failed to list BGM files: {e}")
            return []
    
    def _trim_video_to_duration(self, video: str, target_duration: float) -> str:
        """
        Trim video to specified duration
        
        Args:
            video: Input video file path
            target_duration: Target duration in seconds
        
        Returns:
            Path to trimmed video (temp file)
        
        Raises:
            RuntimeError: If FFmpeg execution fails
        """
        output = self._get_unique_temp_path("trimmed", os.path.basename(video))
        
        try:
            # Use stream copy when possible for fast trimming
            input_stream = ffmpeg.input(video, t=target_duration)
            output_kwargs = {"vcodec": "copy"}
            if self.has_audio_stream(video):
                output_kwargs["acodec"] = "copy"
            (
                input_stream
                .output(output, **output_kwargs)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            return output
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error trimming video: {error_msg}")
            raise RuntimeError(f"Failed to trim video: {error_msg}")
    
    def _pad_video_to_duration(self, video: str, target_duration: float, pad_strategy: str = "freeze") -> str:
        """
        Pad video to specified duration by extending the last frame or adding black frames
        
        Args:
            video: Input video file path
            target_duration: Target duration in seconds
            pad_strategy: Padding strategy - "freeze" (freeze last frame) or "black" (black screen)
        
        Returns:
            Path to padded video (temp file)
        
        Raises:
            RuntimeError: If FFmpeg execution fails
        """
        output = self._get_unique_temp_path("padded", os.path.basename(video))
        
        video_duration = self._get_video_duration(video)
        pad_duration = target_duration - video_duration
        
        if pad_duration <= 0:
            # No padding needed, return original
            return video
        
        try:
            input_video = ffmpeg.input(video)
            video_stream = input_video.video
            
            if pad_strategy == "freeze":
                # Freeze last frame using tpad filter
                video_stream = video_stream.filter('tpad', stop_mode='clone', stop_duration=pad_duration)
                
                # Output with re-encoding (tpad requires it)
                (
                    ffmpeg
                    .output(
                        video_stream,
                        output,
                        vcodec='libx264',
                        preset='fast',
                        crf=23
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
            else:  # black
                # Generate black frames for padding duration
                # Get video properties
                probe = ffmpeg.probe(video)
                video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                width = int(video_info['width'])
                height = int(video_info['height'])
                fps_str = video_info['r_frame_rate']
                fps_num, fps_den = map(int, fps_str.split('/'))
                fps = fps_num / fps_den if fps_den != 0 else 30
                
                # Create black video for padding
                black_input = ffmpeg.input(
                    f'color=c=black:s={width}x{height}:r={fps}',
                    f='lavfi',
                    t=pad_duration
                )
                
                # Concatenate original video with black padding
                video_stream = ffmpeg.concat(video_stream, black_input.video, v=1, a=0)
                
                (
                    ffmpeg
                    .output(
                        video_stream,
                        output,
                        vcodec='libx264',
                        preset='fast',
                        crf=23
                    )
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
            
            return output
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error padding video: {error_msg}")
            raise RuntimeError(f"Failed to pad video: {error_msg}")

