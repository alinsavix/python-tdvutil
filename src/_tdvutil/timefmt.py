import time


# Convert seconds to HH:MM:SS.SSS format. Sure, this could use strftime
# or datetime.timedelta, but both of those have their own issues when
# you want a consistent format involving milliseconds.
def sec_to_hms(secs: float, use_ms: bool = True, use_hours: bool = True) -> str:
    """
    Simple conversion from a time in seconds, to hh:mm:ss.sss format

    :param secs: A length of time to convert, in seconds
    :type secs: float
    :param use_ms: include milliseconds in the output
    :type use_ms: bool
    :param use_hours: include hours digits in the output, even if zero
    :type use_hours: bool

    :return: A string in the format of [hh]:mm:ss[.sss]
    :rtype: str
    """
    hours = int(secs // (60 * 60))
    secs %= (60 * 60)

    minutes = int(secs // 60)
    secs %= 60

    ms = int((secs % 1) * 1000)
    secs = int(secs)

    ms_str = f".{ms:03d}" if use_ms else ""
    hour_str = f"{hours:02d}:" if (use_hours or hours > 0) else ""

    return f"{hour_str}{minutes:02d}:{secs:02d}{ms_str}"


# Convert seconds to a SMPTE non-drop-frame timecode string in the format of
# HH:MM:SS:FF

def sec_to_timecode(secs: float, fps: float, dropframe: bool = False) -> str:
    """
    Converts a time in seconds to a timecode string in the format HH:MM:SS:FF.

    :param secs: The length of time to convert, in seconds.
    :type secs: float
    :param fps: The frame rate of the timecode.
    :type fps: float
    :param dropframe: Whether to use drop-frame timecode (not currently supported).
    :type dropframe: bool, default False

    :return: The timecode string in the format HH:MM:SS:FF
    :rtype: str

    :raises ValueError: If drop-frame timecode or non-integer frame rates are used.
    """
    if dropframe:
        raise ValueError("Drop-frame timecode not currently supported")

    if not fps.is_integer():
        raise ValueError("Non-integer frame rates not currently supported")

    hours = int(secs // (60 * 60))
    secs %= (60 * 60)

    minutes = int(secs // 60)
    secs %= 60

    frames = int((secs % 1) * fps)
    secs = int(secs)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"


# Convert seconds to a compressed string, e.g. 1h15m6s
def sec_to_shortstr(secs: float) -> str:
    """
    Simple conversion from a time in seconds, to a compressd string format

    :param secs: A length of time to convert, in seconds
    :type secs: float

    :return: A string in the format of e.g. 1h15m6s
    :rtype: str
    """
    hours = int(secs // (60 * 60))
    secs %= (60 * 60)

    minutes = int(secs // 60)
    secs %= 60

    secs = int(secs)

    if hours:
        return f"{hours:d}h{minutes:d}m{secs:d}s"
    elif minutes:
        return f"{minutes:d}m{secs:d}s"
    else:
        return f"{secs:d}s"


# A very basic HH:MM:SS.SSS format to seconds conversion. We could
# use strptime here, but really, who in their right mind wants to use
# strptime? This is simple enough and straightforward. Also handles the
# case of just specifying some number of seconds without the HH or MM parts.
def hms_to_sec(hms: str) -> float:
    """
    Simple conversion from a time string (hh:mm:ss.sss) to a float time

    :param hms: A string in the format of hh:mm:ss.sss
    :type hms: str

    :return: A time in seconds
    :rtype: float
    """

    timesplit = hms.split(":")

    if len(timesplit) == 3:
        h, m, s = timesplit
    elif len(timesplit) == 2:
        h = "0"
        m, s = timesplit
    elif len(timesplit) == 1:
        h = "0"
        m = "0"
        s = timesplit[0]
    else:
        raise ValueError(f"too many fields ({len(timesplit)}) in hh:mm:ss string 'hms'")

    return (int(h) * 60 * 60) + (int(m) * 60) + float(s)



def timecode_to_sec(timecode: str, fps: float, dropframe: bool = False) -> float:
    """
    Converts a timecode string to seconds.

    Args:
        timecode (str): The timecode string in the format "HH:MM:SS:FF" or "MM:SS:FF".
        fps (float): The frame rate in frames per second.
        dropframe (bool, optional): Whether to use drop-frame timecode. Defaults to False.

    Returns:
        float: The timecode converted to seconds.

    Raises:
        ValueError: If drop-frame timecode is specified or if the frame rate is non-integer.

    """

    if dropframe:
        raise ValueError("Drop-frame timecode not currently supported")

    if not fps.is_integer():
        raise ValueError("Non-integer frame rates not currently supported")

    timesplit = timecode.split(":")

    if len(timesplit) == 4:
        h, m, s, f = timesplit
    elif len(timesplit) == 3:
        h = "0"
        m, s, f = timesplit
    elif len(timesplit) == 2:
        h = "0"
        m = "0"
        s, f = timesplit
    else:
        raise ValueError(f"too many or too few fields in timecode '{timecode}")

    return (int(h) * 60 * 60) + (int(m) * 60) + int(s) + (int(f) / fps)
