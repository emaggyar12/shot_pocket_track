import pandas as pd
import datetime as dt

def _is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return pd.isna(x) or (isinstance(x, str) and x.strip() == "")
    except Exception:
        return False

def _to_total_seconds(val) -> int | None:
    """
    Parse many time representations into total seconds since 00:00:00.
    Returns None if val can't be parsed.
    Accepted:
      - 'HH:MM:SS' or 'MM:SS'
      - digit strings like '945' -> 00:09:45, '13456' -> 01:34:56
      - Excel serials (floats < 1 are fractions of a day; >=1 treated as seconds)
      - pandas/py datetime or time
    """
    if _is_missing(val):
        return None

    # pandas/py datetime or time
    if isinstance(val, (pd.Timestamp, dt.datetime)):
        return val.hour * 3600 + val.minute * 60 + val.second
    if isinstance(val, dt.time):
        return val.hour * 3600 + val.minute * 60 + val.second

    # numeric (not bool)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        v = float(val)
        if v < 1.0:  # Excel-style day fraction
            return int(round((v % 1.0) * 24 * 3600))
        else:        # treat as seconds
            return int(round(v))

    # strings
    s = str(val).strip()

    # 'HH:MM:SS' or 'MM:SS'
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h) * 3600 + int(m) * 60 + int(float(sec))
        elif len(parts) == 2:
            m, sec = parts
            return int(m) * 60 + int(float(sec))
        # fall through if odd format

    # raw digits -> pad to HHMMSS
    if s.isdigit():
        s = s.zfill(6)
        h, m, sec = int(s[0:2]), int(s[2:4]), int(s[4:6])
        return h * 3600 + m * 60 + sec

    return None

def _format_hhmmss(total_seconds: int) -> str:
    # normalize and clamp to 0..(24h-1) if you want wraparound
    total_seconds = int(round(total_seconds))
    if total_seconds < 0:
        total_seconds = 0
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def normalize_times_fill_end(start, end):
    """
    Returns (start_str, end_str) both as 'HH:MM:SS'.
    If 'end' is missing, set end = start + 4 seconds (regardless of start format).
    """
    s_sec = _to_total_seconds(start)
    if s_sec is None:
        # If start is unparseable, return empty strings (or raise, if you prefer)
        return "", ""

    e_sec = _to_total_seconds(end)
    if e_sec is None:
        e_sec = s_sec + 4  # fill missing end as start + 4s

    return _format_hhmmss(s_sec), _format_hhmmss(e_sec)


if __name__ == '__main__':
    df = pd.read_excel('video_parse/louisville.xls')

    df[['start', 'end']] = df.apply(
        lambda r: pd.Series(normalize_times_fill_end(r['start'], r['end'])),
        axis=1
    )
    
    df.to_excel('video_parse/edited_poop.xlsx', index = False)