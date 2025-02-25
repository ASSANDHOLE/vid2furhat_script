import json

from process import extract_speech_text_and_timestamps, extract_facial_expressions

# !/usr/bin/env python3

"""
Demo pipeline for:
1) Detecting big expression changes vs. the last big change
2) Aligning to word boundaries
3) Segmenting text
4) Producing final (text, expression, pause) chunks
"""

import os

BASE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output/")
USE_BASE_OUTPUT_PATH = True
_GET_OUTPUT_PATH = lambda x: os.path.join(BASE_OUTPUT_PATH, x) if USE_BASE_OUTPUT_PATH else x

EXPRESSIONS_TO_REMOVE = [
    "EYE_BLINK_LEFT", "EYE_BLINK_RIGHT",
    "EYE_SQUINT_LEFT", "EYE_SQUINT_RIGHT",
]


def calc_expression_delta(expr_a, expr_b):
    """
    Computes a simple sum of absolute square-differences (0-1) across parameters.
    You may want to handle missing params more gracefully if needed.
    """
    keys = set(expr_a.keys()) | set(expr_b.keys())
    delta = 0.0
    for k in keys:
        v1 = expr_a.get(k, 0.0)
        v2 = expr_b.get(k, 0.0)
        delta += abs(v1 - v2) ** 2
    return delta


def find_gesture_breakpoints(expressions, threshold):
    """
    Instead of comparing consecutive frames, we compare the current expression
    to the last 'big change' expression that triggered a breakpoint.

    expressions: list of (time, dict_of_params), sorted by time
    threshold: float, how large the difference must be to trigger a breakpoint

    returns: list of times where a big change occurs
    """
    if not expressions:
        return []

    breakpoints = []
    last_bc_expr = expressions[0][1]  # expression dict at the first frame
    last_bc_time = expressions[0][0]

    for i in range(1, len(expressions)):
        t, expr = expressions[i]
        delta = calc_expression_delta(expr, last_bc_expr)
        if delta > threshold:
            # We consider t a 'big change' time
            breakpoints.append(t)
            last_bc_expr = expr
            last_bc_time = t

    return breakpoints


def align_breakpoints_to_words(breakpoints, words, max_offset=0.3):
    """
    Snap each breakpoint time to the nearest word boundary (word_end)
    if within max_offset seconds. If not within offset, ignore that breakpoint.

    words: list of (word_text, start_time, end_time)
    breakpoints: list of floats (times)
    max_offset: how close in seconds we must be to snap to a boundary

    returns: sorted list of word boundary times to force breaks
    """
    word_boundaries = [w[2] for w in words]  # each word's end_time
    aligned = set()
    wb_idx = 0

    for bp in breakpoints:
        # Advance wb_idx until we find a boundary >= bp or we run out
        while wb_idx < len(word_boundaries) - 1 and word_boundaries[wb_idx] < bp:
            wb_idx += 1

        # Potential candidates for snapping:
        candidates = []
        if wb_idx > 0:
            candidates.append(word_boundaries[wb_idx - 1])
        if wb_idx < len(word_boundaries):
            candidates.append(word_boundaries[wb_idx])

        # Pick whichever candidate is closer to bp (if within max_offset)
        best_candidate = None
        best_dist = float('inf')
        for c in candidates:
            dist = abs(c - bp)
            if dist < best_dist and dist <= max_offset:
                best_candidate = c
                best_dist = dist

        if best_candidate is not None:
            aligned.add(best_candidate)

    return sorted(list(aligned))


def has_punctuation(word):
    """
    Simple check for punctuation that might indicate a natural break
    (commas, periods, question marks, etc.). You can expand as needed.
    """
    punctuation_marks = [".", ",", "!", "?", ";", ":"]
    return any(word.endswith(p) for p in punctuation_marks)


def segment_text(words, forced_breaks, max_duration=10.0):
    """
    Segment the text based on forced break times (aligned from expression changes),
    punctuation, or exceeding max_duration. Each segment:
      - text
      - start_time
      - end_time
    """
    segments = []
    if not words:
        return segments

    current_segment_words = []
    current_segment_start = words[0][1]  # start_time of the first word

    for i, (w_text, w_start, w_end) in enumerate(words):
        current_segment_words.append(w_text)
        duration_so_far = w_end - current_segment_start

        # Decide if we should break
        if (w_end in forced_breaks) or has_punctuation(w_text) or (duration_so_far > max_duration):
            # finalize this segment
            seg_text = " ".join(current_segment_words).strip()
            segments.append((seg_text, current_segment_start, w_end))

            # start a new segment
            current_segment_words = []
            # next segment starts right after this word
            if i < len(words) - 1:
                current_segment_start = words[i + 1][1]  # next word's start time
            else:
                current_segment_start = w_end

    # leftover
    if current_segment_words:
        seg_text = " ".join(current_segment_words).strip()
        segments.append((seg_text, current_segment_start, words[-1][2]))

    return segments


def pick_segment_expression(segments, expressions):
    """
    For each segment, pick a representative expression.
    E.g. we'll pick the expression closest to the segment start time.

    returns: list of (seg_text, expression_dict, seg_start, seg_end)
    """
    results = []
    expr_idx = 0
    n_expr = len(expressions)

    for (seg_text, seg_start, seg_end) in segments:
        # find expression closest to seg_start
        best_expr = None
        best_dist = float('inf')

        # naive search - or you could do a more efficient approach
        for (t, expr_dict) in expressions:
            dist = abs(t - seg_start)
            if dist < best_dist:
                best_expr = expr_dict
                best_dist = dist

        if best_expr is None and n_expr > 0:
            best_expr = expressions[-1][1]  # fallback to last known expression
        elif best_expr is None:
            best_expr = {}

        results.append((seg_text, best_expr, seg_start, seg_end))

    return results


def expected_speak_duration(word, wpms=150, avg_chars_per_word=4):
    """
    Estimate the time in seconds it takes to say a word.
    wpms: words per minute speaking rate
    avg_chars_per_word: average number of characters per word
    """
    return len(word) / avg_chars_per_word / wpms * 60 * 1.2  # 20% buffer


def compute_pauses(segment_info, pause_threshold=50, long_pause_threshold=400):
    """
    Convert segment info into final tuples of:
       (text_chunk, expression_frames, pause_ms)

    Instead of a single expression dict per segment, we return:
       (text_chunk, [ (start_rel, end_rel, expr_dict) ], pause_ms)

    We merge segments **unless**:
      - The text ends with punctuation.
      - A large **pause** (beyond expected speech duration) occurs.

    pause_threshold: Minimum pause before considering a gap meaningful.
    long_pause_threshold: Defines what qualifies as a **long** pause.
    """
    final_list = []
    expr_frames = []
    current_text = ""
    current_ref_time = 0.0
    is_new = True

    for i in range(len(segment_info) - 1):
        seg_text, seg_expr, seg_start, seg_end = segment_info[i]
        next_start = segment_info[i + 1][2]  # next segment's start_time

        # Compute pause in ms
        gap_ms = int((next_start - seg_end) * 1000)
        if gap_ms < pause_threshold:
            gap_ms = 0  # Ignore small pauses

        # Handle **new chunk initialization**
        if is_new:
            expr_frames = [(0.0, seg_end - seg_start, seg_expr)]
            current_text = seg_text
            current_ref_time = seg_start
        else:
            current_text += " " + seg_text  # Properly merge text
            expr_frames.append((seg_start - current_ref_time, seg_end - current_ref_time, seg_expr))

        # Check if we should **split**
        if has_punctuation(seg_text) or (
                next_start - seg_start - expected_speak_duration(seg_text) >= long_pause_threshold / 1000):
            current_text = current_text.strip()
            current_text = current_text.replace("  ", " ")  # remove double spaces
            final_list.append((current_text.strip(), expr_frames, gap_ms))
            is_new = True  # Reset for new chunk
        else:
            is_new = False  # Continue merging

    # Handle the **last chunk**
    if segment_info:
        seg_text, seg_expr, seg_start, seg_end = segment_info[-1]
        if is_new:
            expr_frames = [(0.0, seg_end - seg_start, seg_expr)]
            final_list.append((seg_text, expr_frames, long_pause_threshold))  # Last chunk gets a long pause
        else:
            current_text += " " + seg_text
            expr_frames.append((seg_start - current_ref_time, seg_end - current_ref_time, seg_expr))
            final_list.append((current_text.strip(), expr_frames, long_pause_threshold))

    return final_list


def generate_script(words, expressions):
    # # Example data
    # words = [
    #     ("I", 0.00, 0.10),
    #     ("went", 0.10, 0.46),
    #     ("to", 0.46, 1.08),
    #     ("Ay", 1.08, 1.66),
    #     ("a", 1.66, 2.00),
    #     ("doctor's", 2.00, 2.90),
    #     ("office,", 2.90, 3.30),
    #     ("and", 3.30, 3.70),
    #     ("it's", 3.70, 4.10),
    #     ("not", 4.10, 4.40),
    #     ("Dr.", 4.40, 4.70),
    #     ("Dickhead", 4.70, 5.40),
    #     ("that", 5.40, 5.80),
    #     ("I", 5.80, 5.90),
    #     ("was", 5.90, 6.10),
    #     ("telling", 6.10, 6.50),
    #     ("you", 6.50, 6.70),
    #     ("about.", 6.70, 7.10),
    # ]
    #
    # # expressions: (time_in_seconds, dict_of_expression_params)
    # expressions = [
    #     (0.00, {"BROW_OUTER_UP_LEFT": 0.1, "MOUTH_FUNNEL": 0.0}),
    #     (0.50, {"BROW_OUTER_UP_LEFT": 0.2, "MOUTH_FUNNEL": 0.2}),
    #     (1.30, {"BROW_OUTER_UP_LEFT": 0.7, "MOUTH_FUNNEL": 0.4}),
    #     (1.60, {"BROW_OUTER_UP_LEFT": 0.9, "MOUTH_FUNNEL": 0.8}),  # big change from last
    #     (2.50, {"BROW_OUTER_UP_LEFT": 0.2, "MOUTH_FUNNEL": 0.1}),  # big change from last
    #     (4.65, {"BROW_OUTER_UP_LEFT": 0.8, "MOUTH_FUNNEL": 0.9}),  # big change
    #     (6.80, {"BROW_OUTER_UP_LEFT": 0.0, "MOUTH_FUNNEL": 0.0}),  # big change
    # ]

    # 1) Find big expression changes vs. the last big-change expression
    threshold = 1.0  # tune as needed
    big_changes = find_gesture_breakpoints(expressions, threshold)

    # 2) Align to nearest word boundary
    forced_breaks = align_breakpoints_to_words(big_changes, words, max_offset=0.3)

    # 3) Segment text
    segments = segment_text(words, forced_breaks, max_duration=10.0)

    # 4) Pick expression for each segment
    seg_expressions = pick_segment_expression(segments, expressions)
    # seg_expressions => list of (seg_text, seg_expr, seg_start, seg_end)

    # 5) Compute pause durations
    final_output = compute_pauses(seg_expressions)
    # final_output => [ (text_chunk, expression_dict, pause_ms), ... ]

    return final_output


def drop_small_val(expr_dict, threshold=0.05):
    """
    Drop small values from the expression dict to reduce noise.
    """
    return {k: (v if v > threshold else 0) for k, v in expr_dict.items()}


def get_script_from_vid(video_path,
                        preloaded_text=None, preloaded_expressions=None,
                        save_expressions=None, save_text=None):
    # Extract speech and timestamps
    if preloaded_text is None:
        sentences, words = extract_speech_text_and_timestamps(video_path)
    else:
        with open(_GET_OUTPUT_PATH(preloaded_text), "r") as f:
            data = json.load(f)
            sentences, words = data["sentences"], data["words"]

    # Extract facial expressions
    if preloaded_expressions is None:
        expressions = extract_facial_expressions(video_path)
        expressions = [(t / 1000, drop_small_val(e)) for t, e in expressions]
    else:
        with open(_GET_OUTPUT_PATH(preloaded_expressions), "r") as f:
            expressions = json.load(f)

    # Save data if needed
    if save_expressions is not None:
        with open(_GET_OUTPUT_PATH(save_expressions), "w") as f:
            json.dump(expressions, f)
    if save_text is not None:
        with open(_GET_OUTPUT_PATH(save_text), "w") as f:
            json.dump({"sentences": sentences, "words": words}, f)

    # Generate script
    script = generate_script(words, expressions)

    return script


def generate_furhat_script(final_chunks, state_name="Acting", parent_state="Parent"):
    """
    Generates a Kotlin Furhat skill script from final_chunks.

    final_chunks: list of (text_chunk, expr_frames, pause_ms)
       e.g. [
         ("I went to Ay", [(0.0, 2.0, {"BROW_OUTER_UP_LEFT": 0.7})], 200),
         ("a doctor's office,", [(0.0, 2.0, {"BROW_OUTER_UP_LEFT": 0.1, "MOUTH_FUNNEL": 0.6})], 500),
         ...
       ]

    - Each expr_frames list maps to a unique defineGesture block with multiple frames.
    - The script calls `furhat.gesture(...)` once per chunk (before speaking).
    - Then `furhat.say(...)` plays the chunk.
    - Finally, a `delay(pause_ms)` occurs before moving to the next chunk.
    """

    # 1) Generate unique gesture names for each chunk
    gesture_index = 1
    chunk_gestures = []

    for text, expr_frames, pause_ms in final_chunks:
        gesture_name = f"exp{gesture_index}"
        chunk_gestures.append((gesture_name, text, expr_frames, pause_ms))
        gesture_index += 1

    # 2) Build gesture definitions (multi-frame per chunk)
    gesture_defs = []
    for gesture_name, _, expr_frames, _ in chunk_gestures:
        lines = []
        lines.append(f"val {gesture_name} = defineGesture {{")

        for start_rel_time, end_rel_time, expr_dict in expr_frames:
            # Calculate frame duration (difference between current and next time)
            duration = max(0.1, end_rel_time - start_rel_time)

            lines.append(f"    frame({start_rel_time:.2f}, {duration:.2f}) {{")
            for param, value in expr_dict.items():
                if param not in EXPRESSIONS_TO_REMOVE:
                    lines.append(f"        {param} to {value}")
            lines.append("    }")

        lines.append("}")
        gesture_defs.append("\n".join(lines))

    gesture_section = "\n\n".join(gesture_defs)

    # 3) Build the Furhat state logic
    state_lines = []
    state_lines.append(f"val {state_name}: State = state({parent_state}) {{")
    state_lines.append("    onEntry {")

    for gesture_name, text, _, pause_ms in chunk_gestures:
        safe_text = text.replace('"', '\\"')

        state_lines.append(f"        furhat.gesture({gesture_name})")
        state_lines.append(f"        furhat.say(\"{safe_text}\")")
        if pause_ms > 0:
            state_lines.append(f"        delay({pause_ms})")

    state_lines.append("    }")
    state_lines.append("}")

    state_section = "\n".join(state_lines)

    # 4) Return the two sections as a tuple
    return state_section, gesture_section


def main():
    global USE_BASE_OUTPUT_PATH
    USE_BASE_OUTPUT_PATH = True
    video_path = "./st2_cut.mp4"
    # final_chunks = get_script_from_vid(video_path, save_expressions="expressions.json", save_text="text.json")
    final_chunks = get_script_from_vid(video_path, preloaded_text="text.json", preloaded_expressions="expressions.json")
    state, gesture = generate_furhat_script(final_chunks)

    with open("output/state.kt", "w") as f:
        f.write(state)
    with open("output/gesture.kt", "w") as f:
        f.write(gesture)


if __name__ == "__main__":
    main()
