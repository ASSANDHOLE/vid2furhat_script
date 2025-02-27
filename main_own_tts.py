import json
import re
from difflib import SequenceMatcher

from mediapipe.python.packet_getter import get_bytes

from main import drop_small_val
from process import extract_speech_text_and_timestamps, extract_facial_expressions
import numpy as np
from scipy.interpolate import interp1d
import os
from copy import deepcopy

BASE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output/")
USE_BASE_OUTPUT_PATH = True
_GET_OUTPUT_PATH = lambda x: os.path.join(BASE_OUTPUT_PATH, x) if USE_BASE_OUTPUT_PATH else x

EXPRESSIONS_TO_REMOVE = [
    "EYE_BLINK_LEFT", "EYE_BLINK_RIGHT",
    "EYE_SQUINT_LEFT", "EYE_SQUINT_RIGHT",
    # "MOUTH_CLOSE"
]

EOS = "<eos>"

def align_words_lcs(original_words, new_words):
    """
    Align words using LCS, returning segments for time alignment.
    """
    matcher = SequenceMatcher(None, original_words, new_words)
    lcs_result = []

    for tag, orig_i1, orig_i2, new_i1, new_i2 in matcher.get_opcodes():
        lcs_result.append((tag, orig_i1, orig_i2, new_i1, new_i2))

    return lcs_result


def get_lcs_time_mapping(original_timestamps, new_timestamps, end_delta):
    """
    Preprocess LCS results to generate time mapping for each segment.
    timestamps: [(word, start, end), ...]
    """
    original_words = [t.strip() for t, _, _ in original_timestamps]
    new_words = [t.strip() for t, _, _ in new_timestamps]
    lcs_result = align_words_lcs(original_words, new_words)
    time_mapping = []
    last_end = last_new_end = 0

    def m_trans(start_orig, end_orig, start_new, end_new):
        dur_orig = end_orig - start_orig
        dur_new = end_new - start_new
        assert dur_orig > 0 and dur_new > 0
        return lambda original_time: start_new + (original_time - start_orig) * dur_new / dur_orig

    for tag, orig_i1, orig_i2, new_i1, new_i2 in lcs_result:
        if tag == "equal":
            for o1, n1 in zip(original_timestamps[orig_i1:orig_i2], new_timestamps[new_i1:new_i2]):
                o_s, o_e, n_s, n_e = o1[1], o1[2], n1[1], n1[2]
                if last_end is not None:
                    gap_orig = o_s - last_end
                    gap_new = n_s - last_new_end
                    if gap_orig > 0 and gap_new > 0:
                        time_mapping.append((last_end, o_s, m_trans(last_end, o_s, last_new_end, n_s)))
                    else:
                        o_s, n_s = last_end, last_new_end
                time_mapping.append((o_s, o_e, m_trans(o_s, o_e, n_s, n_e)))
                last_end, last_new_end = o_e, n_e
        elif tag == "replace":
            o_s = original_timestamps[orig_i1][1]
            o_e = original_timestamps[orig_i2 - 1][2]
            n_s = new_timestamps[new_i1][1]
            n_e = new_timestamps[new_i2 - 1][2]
            if last_end is not None:
                o_s, n_s = last_end, last_new_end
            time_mapping.append((o_s, o_e, m_trans(o_s, o_e, n_s, n_e)))
            last_end, last_new_end = o_e, n_e
        elif tag == "delete" or tag == "insert":
            pass  # Handled by equal/replace with last known end times

    end_delta = max(end_delta, 0.02)
    time_mapping.append((last_end, last_end + end_delta, lambda x: x + last_new_end - last_end))
    return time_mapping


def resample_expressions(expression_data, target_fps=5, extrapolate=True):
    """
    Resamples expression data to a fixed FPS, handling missing periods smoothly.

    expression_data: [(time, expression_dict), ...]  # Sorted list of (time, expressions)
    target_fps: Target frames per second.
    extrapolate: If True, fill missing frames with last known expression.

    Returns: [(new_time, expression_dict), ...]  # Smoothed expressions
    """
    if not expression_data:
        return []

    # Extract timestamps & expression keys
    times = np.array([t for t, _ in expression_data])
    min_time, max_time = times[0], times[-1]

    # Generate resampled timestamps (uniform intervals at target_fps)
    target_times = np.arange(min_time, max_time, 1.0 / target_fps)

    # Get all unique expression keys
    expression_keys = set()
    for _, expr_dict in expression_data:
        expression_keys.update(expr_dict.keys())

    # Prepare interpolation functions for each expression parameter
    interpolated_expressions = []
    interpolators = {}

    for key in expression_keys:
        values = np.array([expr.get(key, 0.0) for _, expr in expression_data])

        # Use linear interpolation, with optional extrapolation
        interp_func = interp1d(
            times, values, kind='linear',
            fill_value="extrapolate" if extrapolate else 0.0
        )
        interpolators[key] = interp_func

    # Generate interpolated values for each target timestamp
    for new_time in target_times:
        new_expr_dict = {key: interpolators[key](new_time) for key in expression_keys}
        interpolated_expressions.append((new_time, new_expr_dict))

    return interpolated_expressions


def time_mapping_pipeline(original_timestamps, new_timestamps, expressions, end_deltas, start_times, target_fps=5):
    assert len(original_timestamps) == len(new_timestamps) == len(end_deltas) == len(start_times)
    res = []
    for i in range(len(original_timestamps)):
        time_mapping = get_lcs_time_mapping(original_timestamps[i], new_timestamps[i], end_deltas[i])
        expression_in_range = [(t - start_times[i], e) for t, e in expressions if start_times[i] <= t < start_times[i]+time_mapping[-1][1]]
        mapped_expression = []
        for m in time_mapping:
            for t, e in expression_in_range:
                if m[0] <= t < m[1]:
                    mapped_expression.append((m[2](t), e))
        resampled_expression = resample_expressions(mapped_expression, target_fps=target_fps)
        res.append(resampled_expression)
    return res


def load_tts_words_file(file_dir):
    with open(file_dir, "r") as f:
        return json.load(f)


def split_original_words(sentences, words):
    """
    Split original [(word, start, end), ...] into
    [[(word, start, end), ...], ...] for each sentence.
    """
    res = []
    end_deltas = []
    cur_sentence = []
    words = deepcopy(words)
    for i, sen in enumerate(sentences[:-1]):
        cur_start_time = sen[1]
        while words and words[0][2] <= sen[2]:
            word = words.pop(0)
            word[1] -= cur_start_time
            word[2] -= cur_start_time
            if word[1] == word[2]:
                continue
            cur_sentence.append(word)
        if not cur_sentence:
            raise ValueError("No words found for sentence.")
        res.append(cur_sentence)
        cur_sentence = []
        end_deltas.append(sentences[i + 1][1] - sen[2])
    if not words:
        raise ValueError("No words found for last sentence.")
    cur_start_time = sentences[-1][1]
    for w in words:
        w[1] -= cur_start_time
        w[2] -= cur_start_time
    res.append(words)
    end_deltas.append(1.0)
    return res, end_deltas, [s[1] for s in sentences]  # [start_time, ...]


def generate_furhat_script(sentences, mapped_expressions, end_deltas, url, state_name="Acting", parent_state="Parent"):
    # 1) Generate unique gesture names for each clip
    gesture_defs = []
    state_lines = []
    state_lines.append(f"val {state_name}: State = state({parent_state}) {{")
    state_lines.append("    onEntry {")

    for index, ((text, start, end), expr_frames) in enumerate(zip(sentences, mapped_expressions)):
        gesture_name = f"exp{index}"
        audio_url = url.format(index)
        safe_text = text.replace('"', '\\"')

        # Generate single defineGesture block for the whole clip
        gesture_lines = [f"val {gesture_name} = defineGesture {{"]
        prev_duration = None

        for i, (t_start, expr_dict) in enumerate(expr_frames):
            t_end = expr_frames[i + 1][0] if i + 1 < len(expr_frames) else (prev_duration + t_start if prev_duration is not None else end - start)
            prev_duration = t_end - t_start
            gesture_lines.append(f"    frame({t_start:.2f}, {t_end:.2f}) {{")
            for param, value in expr_dict.items():
                if param not in EXPRESSIONS_TO_REMOVE:
                    values_str = f"{value:.4f}"
                    # reduce number of decimals if end is 0
                    if values_str.endswith(".0000"):
                        values_str = values_str[:-5]
                    gesture_lines.append(f"        {param} to {values_str}")
            gesture_lines.append("    }")

        gesture_lines.append("}")
        gesture_defs.append("\n".join(gesture_lines))

        # Add state execution sequence
        state_lines.append(f"        furhat.gesture({gesture_name})")
        state_lines.append(f"        furhat.say{{+Audio(\"{audio_url}\", \"{safe_text}\", speech = true)}}")
        state_lines.append(f"        delay({max(int(end_deltas[index] * 1000), 20)})")

    state_lines.append("    }")
    state_lines.append("}")

    # Final combined script
    return "\n".join(state_lines), "\n\n".join(gesture_defs)


def get_script_from_vid(video_path,
                        preloaded_text=None, preloaded_expressions=None,
                        tts_words_dir=None, tts_words_naming="output_{}.json",
                        save_expressions=None, save_text=None,
                        url="http://127.0.0.1:8888/output_{}.wav"):
    # Extract speech and timestamps
    if preloaded_text is None:
        sentences, words = extract_speech_text_and_timestamps(video_path)
    else:
        with open(_GET_OUTPUT_PATH(preloaded_text), "r") as f:
            data = json.load(f)
            sentences, words = data["sentences"], data["words"]

    if save_text is not None:
        with open(_GET_OUTPUT_PATH(save_text), "w") as f:
            json.dump({"sentences": sentences, "words": words}, f)

    # Extract facial expressions
    if preloaded_expressions is None:
        expressions = extract_facial_expressions(video_path)
        expressions = [(t / 1000, drop_small_val(e)) for t, e in expressions]
    else:
        with open(_GET_OUTPUT_PATH(preloaded_expressions), "r") as f:
            expressions = json.load(f)

    if save_expressions is not None:
        with open(_GET_OUTPUT_PATH(save_expressions), "w") as f:
            json.dump(expressions, f)



    if tts_words_dir is None:
        raise ValueError("tts_words dir path must be provided.")
    files = os.listdir(tts_words_dir)
    regex_pattern = tts_words_naming.replace("{}", r"(\d+)")
    pattern = re.compile(f'^{regex_pattern}$')
    matches = [(s, int(pattern.match(s).group(1))) for s in files if pattern.match(s)]
    matches.sort(key=lambda x: x[1])
    tts_words_data = []
    for file, idx in matches:
        data = load_tts_words_file(os.path.join(tts_words_dir, file))
        tts_words_data.append(data)
    words_list, end_deltas, start_times = split_original_words(sentences, words)
    res = time_mapping_pipeline(words_list, tts_words_data, expressions, end_deltas, start_times)
    scripts = generate_furhat_script(sentences, res, end_deltas,
                            url=url)

    return scripts


def main():
    video_path = "./st2_cut.mp4"
    tts_words_dir = "./TTS/resources/out/"
    preloaded_text = "./text.json"
    preloaded_expressions = "./expressions.json"
    # url = "https://s3-eu-west-1.amazonaws.com/furhat-users/b1344330-fc7d-4fc0-b07e-3c44bd913b5a/audio/output_{}.wav"
    url = "classpath:audio/output_{}.wav"
    # url = "http://127.0.0.1:8888/output_{}.wav"
    script, gestures = get_script_from_vid(video_path, preloaded_text=preloaded_text, preloaded_expressions=preloaded_expressions,
                                           tts_words_dir=tts_words_dir,
                                           url=url)
    with open(_GET_OUTPUT_PATH("state_v2.kt"), "w") as f:
        f.write(script)
    with open(_GET_OUTPUT_PATH("gestures_v2.kt"), "w") as f:
        f.write(gestures)


if __name__ == "__main__":
    main()
