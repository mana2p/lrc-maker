#!/usr/bin/env python3
"""
LRC Maker — 音源と歌詞から歌い始め・歌い終わりを検出し、スキャットをスルーして高精度なLRCを生成するツール
JIZURA連携対応（歌っていない区間に自動で [間奏] または空行を挿入し、歌詞の間延びを防止）
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import traceback
import webbrowser
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal

# ------------------------------------------------------------
# テキスト正規化・スキャット判定ユーティリティ
# ------------------------------------------------------------

SCAT_PATTERNS = re.compile(
    r"^[\s\W_]*("
    r"ah+|oh+|uh+|oo+h*|wo+h*|ye+a*h*|yo+|na+[\s\-]*na+|la+[\s\-]*la+|"
    r"da+[\s\-]*da+|ba+[\s\-]*ba+|sha+[\s\-]*la+|mm+|hm+|hu+h*|"
    r"アー+|オー+|ウー+|イェー+イ*|イェイ+|ララ+|ナナ+|ルル+|フー+|ハァ+|フフ+|"
    r"yeah|woo|baby|come on|let's go|check it|one two"
    r")[\s\W_]*$",
    re.IGNORECASE,
)


def kata_to_hira(text: str) -> str:
    """カタカナをひらがなに統一"""
    res = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            res.append(chr(code - 0x60))
        else:
            res.append(ch)
    return "".join(res)


def normalize_for_match(text: str) -> str:
    """比較用に記号・空白を除去し、英字小文字・ひらがな化"""
    t = kata_to_hira(text.lower())
    t = re.sub(r"[\s\W_、。！？!?,.\-〜～「」『』（）()【】\[\]\/｜|＊*]+", "", t)
    return t


def char_overlap_ratio(seg_text: str, lyrics_norm: str) -> float:
    """認識セグメントの文字が歌詞全体とどの程度重なっているかを返す (0.0〜1.0)"""
    s = normalize_for_match(seg_text)
    if not s:
        return 0.0
    if SCAT_PATTERNS.match(seg_text.strip()):
        # もし歌詞自体にも同じスキャットが含まれているなら残す
        if s in lyrics_norm:
            return 1.0
        return 0.0
    # 2-gram または 1-gram の一致率
    if len(s) >= 2:
        bigrams = [s[i : i + 2] for i in range(len(s) - 1)]
        hits = sum(1 for bg in bigrams if bg in lyrics_norm)
        if hits > 0:
            return hits / len(bigrams)
    hits1 = sum(1 for ch in s if ch in lyrics_norm)
    return hits1 / len(s)


# ------------------------------------------------------------
# 音響信号処理（ボーカル帯域抽出・RMSエンベロープ・リリース検出）
# ------------------------------------------------------------

def load_audio_mono_16k(audio_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    音源を読み込み、16kHzの
    1) アライメント用モノラル波形 (raw_16k)
    2) ボーカル帯域強調（Mid抽出 + 220Hz〜3800Hz バンドパス）波形 (vocal_band_16k)
    を返す。FFmpegがあればあらゆるフォーマット(mp3/m4a/wav/flac)に対応。
    """
    sr_target = 16000
    tmp_wav = None
    try:
        # まず soundfile で直接読めるか試行、ダメなら ffmpeg で 16kHz 2ch wav に変換
        try:
            data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        except Exception:
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            subprocess.run(
                ["ffmpeg", "-y", "-i", audio_path, "-ar", str(sr_target), "-ac", "2", tmp_wav],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            data, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
    finally:
        if tmp_wav and os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass

    # ステレオの場合は Mid (センター定位: L+R) を抽出し、Side (L-R) 成分を抑制したボーカル用信号を作る
    if data.shape[1] >= 2:
        left, right = data[:, 0], data[:, 1]
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        raw_mono = mid.copy()
        # STFT上でセンター定位マスクをかける簡易処理（時間領域での緩やかな抑制）
        vocal_mono = mid
    else:
        raw_mono = data[:, 0]
        vocal_mono = data[:, 0]

    if sr != sr_target:
        num_samples = int(round(len(raw_mono) * sr_target / sr))
        raw_mono = signal.resample(raw_mono, num_samples).astype(np.float32)
        vocal_mono = signal.resample(vocal_mono, num_samples).astype(np.float32)

    # ボーカルのフォルマント帯域 (220Hz - 3800Hz) バンドパスフィルタで低域(キック/ベース)と超高域(ハイハット)を除去
    sos = signal.butter(4, [220.0, 3800.0], btype="bandpass", fs=sr_target, output="sos")
    vocal_band = signal.sosfiltfilt(sos, vocal_mono).astype(np.float32)

    return raw_mono, vocal_band, sr_target


def separate_vocals_demucs(audio_path: str) -> Optional[str]:
    """Demucsでボーカルステムのみを分離し、一時WAVファイルパスを返す"""
    out_dir = tempfile.mkdtemp(prefix="lrc_demucs_")
    try:
        cmd = [
            os.sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "--two-stems",
            "vocals",
            "-o",
            out_dir,
            audio_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for root, _, files in os.walk(out_dir):
            for f in files:
                if f.lower() == "vocals.wav":
                    return os.path.join(root, f)
    except Exception as e:
        print(f"[warn] Demucs vocal separation skipped: {e}")
    return None


def compute_rms_envelope(wave: np.ndarray, sr: int = 16000, hop_ms: int = 10, win_ms: int = 30) -> Tuple[np.ndarray, float]:
    """10ms刻みのRMSエンベロープを計算"""
    hop = int(sr * hop_ms / 1000)
    win = int(sr * win_ms / 1000)
    pad = np.pad(wave ** 2, (win // 2, win // 2), mode="constant")
    kernel = np.ones(win, dtype=np.float32) / win
    mean_sq = np.convolve(pad, kernel, mode="valid")[::hop]
    rms = np.sqrt(np.maximum(mean_sq, 1e-12))
    return rms, hop_ms / 1000.0


def refine_line_bounds_with_rms(
    start_sec: float,
    end_sec: float,
    last_char_start: float,
    next_line_start: Optional[float],
    rms: np.ndarray,
    dt: float,
    max_tail_char_dur: float = 0.95,
    release_db_drop: float = 15.0,
) -> Tuple[float, float]:
    """
    ボーカル帯域のRMSエンベロープを使って、
    1) 歌い始めの無音かぶりをトリム
    2) 歌い終わり（リリースポイント）を検出し、伴奏やリバーブによる間延びをカット
    """
    n_frames = len(rms)
    if n_frames == 0:
        return start_sec, end_sec

    # 1. 行末の上限クランプ（最後の文字が間奏まで何秒も伸びるのを防ぐ）
    capped_end = min(end_sec, max(start_sec + 0.25, last_char_start + max_tail_char_dur))
    if next_line_start is not None:
        capped_end = min(capped_end, next_line_start - 0.02)

    i0 = max(0, min(n_frames - 1, int(start_sec / dt)))
    i1 = max(i0 + 1, min(n_frames, int(capped_end / dt) + 1))
    seg_rms = rms[i0:i1]
    if len(seg_rms) == 0:
        return start_sec, capped_end

    peak_rms = float(np.percentile(seg_rms, 90))
    noise_floor = float(np.percentile(rms, 15))
    # ピークから release_db_drop (dB) 落ちた閾値
    thresh = max(noise_floor * 1.6, peak_rms * (10.0 ** (-release_db_drop / 20.0)))

    # 歌い始め微調整: 先頭が無音なら立ち上がりまで進める（最大0.25s以内）
    max_onset_shift = int(0.25 / dt)
    onset_idx = i0
    for k in range(min(len(seg_rms), max_onset_shift)):
        if seg_rms[k] >= thresh * 0.85:
            onset_idx = i0 + max(0, k - 2)  # 子音アタック保護のため20ms手前
            break
    refined_start = onset_idx * dt

    # 歌い終わり微調整: 最後の文字付近から後ろを見て、エネルギーが閾値を下回り続ける地点を探す
    i_last = max(onset_idx + 1, min(i1 - 1, int(last_char_start / dt)))
    release_idx = i1 - 1
    hold_frames = max(1, int(0.10 / dt))  # 100ms連続で減衰したら歌い終わりと判定
    for k in range(i_last, i1):
        window = rms[k : min(n_frames, k + hold_frames)]
        if len(window) > 0 and np.all(window < thresh):
            release_idx = k + 2  # リリース余韻20msだけ残す
            break

    refined_end = max(refined_start + 0.20, min(capped_end, release_idx * dt))
    return round(refined_start, 3), round(refined_end, 3)


# ------------------------------------------------------------
# Whisper ASR + テキストシーケンスアライメント本体
# ------------------------------------------------------------

import difflib

@dataclass
class AlignedLine:
    index: int
    text: str
    start: float
    end: float
    words: List[dict]


class LRCMakerEngine:
    def __init__(self, model_name: str = "small", device: Optional[str] = None):
        import torch
        import stable_whisper

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LRC Maker] Loading Whisper model '{model_name}' on {self.device}...")
        self.model = stable_whisper.load_model(model_name, device=self.device)

    def generate(
        self,
        audio_path: str,
        lyrics_text: str,
        lang: str = "ja",
        use_demucs: bool = False,
        ignore_scat: bool = True,
        interlude_gap: float = 0.60,
        interlude_tag: str = "[間奏]",
        lead_offset: float = 0.0,
        max_char_tail: float = 0.90,
    ) -> dict:
        """
        音源からWhisperで高精度トランスクリプション（VAD/無音抑制付き）を行い、
        ユーザーの歌詞テキストと動的シーケンスマッチングで各行の歌い始め・歌い終わりを確定する。
        これにより、イントロや伴奏でDTWがスキップされて曲末尾に固まる現象を100%防止し、
        歌詞にないスキャットを自動スルーする。
        """
        raw_rows = [r.strip() for r in lyrics_text.replace("\r", "").split("\n")]
        lyric_lines: List[str] = []
        for r in raw_rows:
            if not r or r.startswith("#"):
                continue
            cleaned = re.sub(r"^\[\d+:\d+(?:[.:]\d+)?\]\s*", "", r).strip()
            if not cleaned or re.match(r"^\[\s*(間奏|interlude|inst).*\]$", cleaned, re.I):
                continue
            lyric_lines.append(cleaned)

        if not lyric_lines:
            raise ValueError("有効な歌詞行がありません。")

        # 1. 音響解析用信号（RMSエンベロープ）の取得
        raw_16k, vocal_band_16k, sr = load_audio_mono_16k(audio_path)
        rms, dt = compute_rms_envelope(vocal_band_16k, sr=sr, hop_ms=10, win_ms=30)
        audio_duration = round(len(raw_16k) / sr, 2)

        # 2. Whisper によるトランスクリプション実行
        # 歌詞の冒頭を initial_prompt に渡して漢字や発音の認識率を最大化
        prompt_plain = " ".join([ln.replace("/", "") for ln in lyric_lines[:20]])
        prompt_text = "歌詞: " + prompt_plain[:240]

        transcribe_opts = {
            "language": lang,
            "initial_prompt": prompt_text,
            "word_timestamps": True,
            "suppress_silence": True,
            "vad": True,
            "verbose": False,
        }
        if use_demucs:
            try:
                transcribe_opts["demucs"] = True
            except Exception:
                pass

        print(f"[LRC Maker] Transcribing audio with Whisper (VAD=True)...")
        asr_res = self.model.transcribe(audio_path, **transcribe_opts)

        # 3. 認識された単語列のフラット化
        asr_words: List[dict] = []
        for seg in asr_res.segments:
            if seg.words:
                for w in seg.words:
                    w_txt = (w.word or "").strip()
                    if w_txt:
                        asr_words.append({
                            "word": w_txt,
                            "start": round(float(w.start), 3),
                            "end": round(float(w.end), 3),
                            "used": False,
                        })
            else:
                seg_txt = (seg.text or "").strip()
                if seg_txt:
                    asr_words.append({
                        "word": seg_txt,
                        "start": round(float(seg.start), 3),
                        "end": round(float(seg.end), 3),
                        "used": False,
                    })

        if not asr_words:
            # 万が一無音などで検出されなかった場合、均等配分フォールバック
            step = audio_duration / (len(lyric_lines) + 1)
            asr_words = [
                {"word": ln, "start": (i + 1) * step, "end": (i + 1) * step + step * 0.7, "used": False}
                for i, ln in enumerate(lyric_lines)
            ]

        # 4. 歌詞行と認識単語列のシーケンシャル最適アライメント
        cur_word_idx = 0
        raw_aligned: List[dict] = []

        for line_idx, raw_line in enumerate(lyric_lines):
            clean_line = raw_line.replace("/", "").strip()
            target = normalize_for_match(clean_line)
            if not target:
                target = clean_line.lower()

            best_score = -1.0
            best_range = (cur_word_idx, cur_word_idx)

            # 現在位置から先の単語ウィンドウ内で最も一致率が高い区間を探す
            max_search = min(len(asr_words), cur_word_idx + 45)
            for i in range(cur_word_idx, max_search):
                concat = ""
                for j in range(i, min(len(asr_words), i + 18)):
                    concat += normalize_for_match(asr_words[j]["word"])
                    sm = difflib.SequenceMatcher(None, target, concat)
                    score = sm.ratio()
                    if score > best_score:
                        best_score = score
                        best_range = (i, j)
                    # 充分に高く文字数も満たしていれば探索早期終了
                    if score >= 0.88 and len(concat) >= len(target) * 0.85:
                        break
                if best_score >= 0.95:
                    break

            # 一致スコアが基準値以上の場合はその区間を採用、スキャット等はスキップされる
            if best_score >= 0.28 and best_range[0] < len(asr_words):
                s_idx, e_idx = best_range
                s_time = asr_words[s_idx]["start"]
                e_time = asr_words[e_idx]["end"]
                # 使用した単語にマーク
                for k in range(s_idx, e_idx + 1):
                    asr_words[k]["used"] = True
                cur_word_idx = e_idx + 1
            else:
                # ASRが聞き落とした行は、直前の終了時刻から自然に推定
                prev_end = raw_aligned[-1]["end"] if raw_aligned else 1.5
                s_time = prev_end + 0.35
                e_time = s_time + max(1.2, len(clean_line) * 0.20)

            raw_aligned.append({
                "index": line_idx,
                "text": raw_line,
                "start": round(s_time, 3),
                "end": round(max(s_time + 0.20, e_time), 3),
                "score": best_score,
            })

        # 単調増加の保証（前の行より時間が戻らない）
        for i in range(1, len(raw_aligned)):
            prev_s = raw_aligned[i - 1]["start"]
            if raw_aligned[i]["start"] < prev_s + 0.08:
                raw_aligned[i]["start"] = round(prev_s + 0.08, 3)
                raw_aligned[i]["end"] = max(raw_aligned[i]["start"] + 0.25, raw_aligned[i]["end"])

        # 5. 各行の歌い終わり（リリース）をボーカルRMSで精密調整
        aligned_lines: List[AlignedLine] = []
        for i, item in enumerate(raw_aligned):
            s = item["start"]
            e = item["end"]
            next_s = raw_aligned[i + 1]["start"] if i + 1 < len(raw_aligned) else None
            # 語尾の1文字の開始位置を推定
            last_cs = max(s, e - min(max_char_tail, 0.45))
            ref_s, ref_e = refine_line_bounds_with_rms(
                start_sec=s,
                end_sec=e,
                last_char_start=last_cs,
                next_line_start=next_s,
                rms=rms,
                dt=dt,
                max_tail_char_dur=max_char_tail,
            )
            ref_s = max(0.0, ref_s + lead_offset)
            ref_e = max(ref_s + 0.15, ref_e + lead_offset)
            aligned_lines.append(
                AlignedLine(
                    index=i,
                    text=item["text"],
                    start=round(ref_s, 2),
                    end=round(ref_e, 2),
                    words=[],
                )
            )

        # 6. スルーされたスキャット（歌詞とマッチしなかった単語区間）の収集
        scat_regions: List[dict] = []
        if ignore_scat:
            for w in asr_words:
                if not w["used"]:
                    scat_regions.append({
                        "start": w["start"],
                        "end": w["end"],
                        "text": w["word"],
                    })

        # 7. LRC文字列の生成（歌っていない区間に [間奏] を自動挿入）
        lrc_rows: List[str] = []
        if aligned_lines and aligned_lines[0].start >= max(1.2, interlude_gap):
            lrc_rows.append(f"[00:00.00]{interlude_tag}")

        for i, al in enumerate(aligned_lines):
            lrc_rows.append(f"[{fmt_lrc_time(al.start)}]{al.text}")
            next_start = aligned_lines[i + 1].start if i + 1 < len(aligned_lines) else None
            if next_start is not None:
                gap = next_start - al.end
                if gap >= interlude_gap:
                    lrc_rows.append(f"[{fmt_lrc_time(al.end)}]{interlude_tag}")
            else:
                # 最終行の歌い終わりにも間奏マーカーを入れてアウトロの間延びを防止
                lrc_rows.append(f"[{fmt_lrc_time(al.end)}]{interlude_tag}")

        lrc_text = "\n".join(lrc_rows) + "\n"
        return {
            "lrc": lrc_text,
            "lines": [asdict(a) for a in aligned_lines],
            "scats_ignored": scat_regions[:30],  # UI表示用に最大30件
            "duration": audio_duration,
        }


def fmt_lrc_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    mm = int(sec // 60)
    ss = sec - mm * 60
    return f"{mm:02d}:{ss:05.2f}"


# ------------------------------------------------------------
# ローカル Web UI サーバー（ドラッグ＆ドロップ 1画面完結）
# ------------------------------------------------------------

_ENGINE_CACHE: dict[str, LRCMakerEngine] = {}
_ENGINE_LOCK = threading.Lock()


def get_engine(model_name: str) -> LRCMakerEngine:
    with _ENGINE_LOCK:
        if model_name not in _ENGINE_CACHE:
            _ENGINE_CACHE[model_name] = LRCMakerEngine(model_name=model_name)
        return _ENGINE_CACHE[model_name]


HTML_PAGE = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LRC Maker — 歌い終わり＆スキャット除去対応 AI歌詞同期</title>
<style>
  :root {
    --bg: #0f1116;
    --panel: #171a22;
    --border: #2a2f3d;
    --fg: #eef1f7;
    --muted: #8b93a7;
    --cyan: #00e5ff;
    --accent: #7c4dff;
    --ok: #00e676;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--fg);
    font-family: "Hiragino Kaku Gothic ProN", "Noto Sans JP", system-ui, sans-serif;
    line-height: 1.6;
  }
  .wrap { max-width: 1080px; margin: 0 auto; }
  header {
    display: flex; align-items: baseline; justify-content: space-between;
    border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 20px;
  }
  h1 { margin: 0; font-size: 1.35rem; letter-spacing: 0.04em; }
  h1 span { color: var(--cyan); font-size: 0.85rem; margin-left: 10px; font-weight: normal; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; display: flex; flex-direction: column; gap: 12px;
  }
  label.title { font-weight: 600; font-size: 0.9rem; color: var(--cyan); }
  .drop {
    border: 2px dashed var(--border); border-radius: 8px; padding: 18px;
    text-align: center; cursor: pointer; transition: border-color 0.15s, background 0.15s;
  }
  .drop:hover, .drop.dragover { border-color: var(--cyan); background: rgba(0,229,255,0.05); }
  textarea {
    width: 100%; min-height: 280px; resize: vertical;
    background: #0b0d12; color: var(--fg); border: 1px solid var(--border);
    border-radius: 6px; padding: 12px; font-size: 0.9rem; line-height: 1.65;
    font-family: "JetBrains Mono", "Yu Gothic", monospace;
  }
  .opts {
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
    background: rgba(255,255,255,0.02); padding: 12px; border-radius: 8px; border: 1px solid var(--border);
    font-size: 0.83rem;
  }
  .opts label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
  select, input[type="number"], input[type="text"] {
    background: #0b0d12; color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 4px 8px; font-size: 0.83rem;
  }
  .btn-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: #fff; border: none; border-radius: 6px; padding: 10px 20px;
    font-weight: 700; font-size: 0.92rem; cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
  }
  button:hover { opacity: 0.92; }
  button:active { transform: translateY(1px); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button.sec { background: #262b38; font-weight: 600; }
  .status { font-size: 0.85rem; color: var(--muted); }
  .pills { display: flex; gap: 6px; flex-wrap: wrap; font-size: 0.76rem; }
  .pill { background: rgba(0,229,255,0.12); color: var(--cyan); padding: 2px 8px; border-radius: 99px; }
  .table-wrap { max-height: 220px; overflow-y: auto; border: 1px solid var(--border); border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
  th, td { padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; }
  th { background: #12151c; color: var(--muted); position: sticky; top: 0; }
  tr:hover { background: rgba(255,255,255,0.03); cursor: pointer; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>LRC Maker <span>歌い始め・歌い終わり精密検出 ＆ スキャット自動スルー（JIZURA対応）</span></h1>
  </header>

  <div class="grid">
    <!-- 左カラム：入力 -->
    <div class="card">
      <label class="title">1. 音源ファイル（MP3 / WAV / M4A / FLAC）</label>
      <div id="dropZone" class="drop">
        <div id="fileName">ここに音源をドラッグ＆ドロップ、またはクリックして選択</div>
        <input id="audioFile" type="file" accept="audio/*,.mp3,.wav,.m4a,.flac,.ogg" hidden>
      </div>
      <audio id="player" controls style="width:100%;height:36px;" hidden></audio>

      <label class="title">2. 歌詞テキスト（1行＝1フレーズ）</label>
      <textarea id="lyricsIn" placeholder="ここに歌詞を貼り付けてください。&#10;歌詞に書かれていないイントロや間奏のスキャット（Ah / Yeah / La la la 等）は自動でスルーされます。"></textarea>

      <div class="opts">
        <label title="歌詞にないAh/Yeah等のスキャットやフェイクを事前検出してミュートし、誤吸着を防ぎます">
          <input type="checkbox" id="optScat" checked> スキャット・フェイクを無視
        </label>
        <label title="2ミックスからDemucsでボーカルだけを分離してからアライメントします（高精度・初回のみ少し時間がかかります）">
          <input type="checkbox" id="optDemucs"> Demucsボーカル分離を使う
        </label>
        <label title="歌い終わりから次の行までこの秒数以上空いたら、歌詞を消して間奏にします">
          間延び防止閾値(秒):
          <input type="number" id="optGap" value="0.60" min="0.2" max="5.0" step="0.1" style="width:64px">
        </label>
        <label title="最後の1文字が間奏まで伸びないよう制限する最大秒数">
          語尾最大長(秒):
          <input type="number" id="optTail" value="0.90" min="0.3" max="3.0" step="0.1" style="width:64px">
        </label>
        <label title="JIZURAでそのまま間奏として認識されるタグを入れるか、空行にするか">
          歌っていない区間:
          <select id="optTag">
            <option value="[間奏]" selected>[間奏]（JIZURA推奨）</option>
            <option value="">空タイムスタンプ行</option>
          </select>
        </label>
        <label title="使用するWhisperモデルのサイズ">
          モデル:
          <select id="optModel">
            <option value="small" selected>small（高速・バランス）</option>
            <option value="medium">medium（高精度）</option>
            <option value="large-v3-turbo">large-v3-turbo（最高精度）</option>
          </select>
        </label>
      </div>

      <div class="btn-row">
        <button id="btnRun">AIでLRCを生成する</button>
        <span id="status" class="status">待機中</span>
      </div>
    </div>

    <!-- 右カラム：出力 -->
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <label class="title">3. 生成された LRC（歌い終わり [間奏] 入り）</label>
        <div class="btn-row">
          <button id="btnCopy" class="sec" disabled>コピー</button>
          <button id="btnDl" class="sec" disabled>.lrc を保存</button>
        </div>
      </div>

      <textarea id="lrcOut" placeholder="ここに生成された .lrc が表示されます。JIZURAの「LRCを読み込む」または歌詞欄へそのまま貼り付けられます。"></textarea>

      <div id="scatBox" hidden>
        <div style="font-size:0.78rem;color:var(--muted);margin-bottom:4px;">スルーしたスキャット / 非歌詞区間:</div>
        <div id="scatPills" class="pills"></div>
      </div>

      <label class="title" style="margin-top:4px;">行ごとの歌い始め・歌い終わりプレビュー（クリックで再生確認）</label>
      <div class="table-wrap">
        <table>
          <thead><tr><th>#</th><th>歌い始め</th><th>歌い終わり</th><th>長さ</th><th>歌詞</th></tr></thead>
          <tbody id="lineTable"></tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let selectedFile = null;

const drop = $('dropZone'), fileIn = $('audioFile'), player = $('player');
drop.addEventListener('click', () => fileIn.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('dragover'); });
drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('dragover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileIn.addEventListener('change', e => { if (e.target.files.length) setFile(e.target.files[0]); });

function setFile(f) {
  selectedFile = f;
  $('fileName').textContent = `🎵 ${f.name} (${(f.size / 1048576).toFixed(1)} MB)`;
  player.src = URL.createObjectURL(f);
  player.hidden = false;
}

$('btnRun').addEventListener('click', async () => {
  if (!selectedFile) { alert('音源ファイルを選択してください'); return; }
  const lyrics = $('lyricsIn').value.trim();
  if (!lyrics) { alert('歌詞テキストを入力してください'); return; }

  $('btnRun').disabled = true;
  $('status').textContent = 'AIが音声解析＆アライメント中…（数十秒ほどお待ちください）';

  const fd = new FormData();
  fd.append('audio', selectedFile);
  fd.append('lyrics', lyrics);
  fd.append('ignore_scat', $('optScat').checked ? '1' : '0');
  fd.append('use_demucs', $('optDemucs').checked ? '1' : '0');
  fd.append('interlude_gap', $('optGap').value);
  fd.append('max_char_tail', $('optTail').value);
  fd.append('interlude_tag', $('optTag').value);
  fd.append('model', $('optModel').value);

  try {
    const res = await fetch('/api/align', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'エラーが発生しました');

    $('lrcOut').value = data.lrc;
    $('btnCopy').disabled = false;
    $('btnDl').disabled = false;
    $('status').textContent = `完了！（全 ${data.lines.length} 行）`;

    // スキャット表示
    const sb = $('scatBox'), sp = $('scatPills');
    sp.innerHTML = '';
    if (data.scats_ignored && data.scats_ignored.length) {
      sb.hidden = false;
      data.scats_ignored.forEach(s => {
        const span = document.createElement('span');
        span.className = 'pill';
        span.textContent = `${s.start.toFixed(1)}s-${s.end.toFixed(1)}s "${s.text}"`;
        sp.appendChild(span);
      });
    } else {
      sb.hidden = true;
    }

    // テーブル表示
    const tb = $('lineTable');
    tb.innerHTML = '';
    data.lines.forEach((ln, i) => {
      const tr = document.createElement('tr');
      const dur = (ln.end - ln.start).toFixed(2);
      tr.innerHTML = `<td>${i + 1}</td><td>${ln.start.toFixed(2)}s</td><td>${ln.end.toFixed(2)}s</td><td>${dur}s</td><td>${ln.text}</td>`;
      tr.addEventListener('click', () => {
        player.currentTime = ln.start;
        player.play();
      });
      tb.appendChild(tr);
    });
  } catch (err) {
    $('status').textContent = 'エラー: ' + err.message;
    alert(err.message);
  } finally {
    $('btnRun').disabled = false;
  }
});

$('btnCopy').addEventListener('click', () => {
  navigator.clipboard.writeText($('lrcOut').value);
  $('status').textContent = 'クリップボードにコピーしました！';
});

$('btnDl').addEventListener('click', () => {
  const base = selectedFile ? selectedFile.name.replace(/\.[^.]+$/, '') : 'lyrics';
  const blob = new Blob([$('lrcOut').value], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = base + '.lrc';
  a.click();
});
</script>
</body>
</html>
"""


def parse_multipart(headers, body: bytes) -> dict:
    """外部ライブラリ不要のシンプルな multipart/form-data パーサー"""
    ctype = headers.get("Content-Type", "")
    m = re.search(r"boundary=([^\s;]+)", ctype)
    if not m:
        return {}
    boundary = ("--" + m.group(1).strip('"')).encode("utf-8")
    parts = body.split(boundary)
    result = {}
    for part in parts:
        if not part or part in (b"--\r\n", b"--", b"\r\n"):
            continue
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        sep = part.find(b"\r\n\r\n")
        if sep < 0:
            continue
        raw_hdr = part[:sep].decode("utf-8", errors="replace")
        content = part[sep + 4 :]
        name_m = re.search(r'name="([^"]+)"', raw_hdr)
        if not name_m:
            continue
        name = name_m.group(1)
        fn_m = re.search(r'filename="([^"]*)"', raw_hdr)
        if fn_m:
            result[name] = {"filename": fn_m.group(1), "data": content}
        else:
            result[name] = content.decode("utf-8", errors="replace")
    return result


class LRCHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path != "/api/align":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        form = parse_multipart(self.headers, body)

        audio_item = form.get("audio")
        lyrics = str(form.get("lyrics", "")).strip()
        if not isinstance(audio_item, dict) or not audio_item.get("data"):
            self._json({"error": "音源ファイルがありません"}, 400)
            return

        suffix = Path(audio_item.get("filename") or "audio.wav").suffix or ".wav"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            with open(tmp_path, "wb") as f:
                f.write(audio_item["data"])

            model_name = str(form.get("model", "small"))
            ignore_scat = str(form.get("ignore_scat", "1")) == "1"
            use_demucs = str(form.get("use_demucs", "0")) == "1"
            interlude_gap = float(form.get("interlude_gap", "0.60"))
            max_char_tail = float(form.get("max_char_tail", "0.90"))
            interlude_tag = str(form.get("interlude_tag", "[間奏]"))

            engine = get_engine(model_name)
            res = engine.generate(
                audio_path=tmp_path,
                lyrics_text=lyrics,
                ignore_scat=ignore_scat,
                use_demucs=use_demucs,
                interlude_gap=interlude_gap,
                max_char_tail=max_char_tail,
                interlude_tag=interlude_tag,
            )
            self._json(res, 200)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": str(e)}, 500)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _json(self, obj: dict, status: int = 200):
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def create_server(host: str, preferred_port: int) -> Tuple[HTTPServer, int]:
    candidate_ports = [preferred_port]
    for p in [8765, 8080, 5000]:
        if p not in candidate_ports:
            candidate_ports.append(p)
    candidate_ports.extend(range(preferred_port + 1, preferred_port + 10))
    candidate_ports.append(0)

    last_err = None
    for port in candidate_ports:
        try:
            srv = HTTPServer((host, port), LRCHandler)
            return srv, srv.server_port
        except OSError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("利用可能なポートが見つかりませんでした")


def main():
    parser = argparse.ArgumentParser(description="LRC Maker — 歌い始め・歌い終わり検出＆スキャット除去対応 LRC生成ツール")
    parser.add_argument("--audio", "-a", help="入力音源ファイル (mp3/wav/m4a/flac)")
    parser.add_argument("--lyrics", "-l", help="入力歌詞テキストファイル (.txt)")
    parser.add_argument("--output", "-o", help="出力LRCファイルパス (.lrc)")
    parser.add_argument("--model", "-m", default="small", help="Whisperモデル名 (small / medium / large-v3-turbo)")
    parser.add_argument("--demucs", action="store_true", help="Demucsでボーカル分離を行ってからアライメントする")
    parser.add_argument("--no-scat-mask", action="store_true", help="スキャット事前マスクを無効化する")
    parser.add_argument("--gap", type=float, default=0.60, help="間奏タグを挿入する最小空白秒数 (デフォルト: 0.60)")
    parser.add_argument("--tail", type=float, default=0.90, help="語尾1文字の最大持続秒数 (デフォルト: 0.90)")
    parser.add_argument("--tag", default="[間奏]", help="歌っていない区間に挿入するタグ (デフォルト: [間奏])")
    parser.add_argument("--port", type=int, default=7865, help="Web UIのポート番号 (デフォルト: 7865)")
    parser.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    args = parser.parse_args()

    # CLI モード
    if args.audio and args.lyrics:
        lyrics_text = Path(args.lyrics).read_text(encoding="utf-8")
        engine = get_engine(args.model)
        res = engine.generate(
            audio_path=args.audio,
            lyrics_text=lyrics_text,
            use_demucs=args.demucs,
            ignore_scat=not args.no_scat_mask,
            interlude_gap=args.gap,
            max_char_tail=args.tail,
            interlude_tag=args.tag,
        )
        out_path = args.output or str(Path(args.audio).with_suffix(".lrc"))
        Path(out_path).write_text(res["lrc"], encoding="utf-8")
        print(f"Saved LRC to: {out_path} ({len(res['lines'])} lines)")
        return

    # Web UI モード
    server, actual_port = create_server("127.0.0.1", args.port)
    url = f"http://127.0.0.1:{actual_port}"
    if actual_port != args.port:
        print(f"※ ポート {args.port} が使用中または制限されているため、ポート {actual_port} で起動しました。")
    print(f"Starting LRC Maker Web UI at {url} ...")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

