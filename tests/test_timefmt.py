from _tdvutil.timefmt import (hms_to_sec, sec_to_hms, sec_to_shortstr,
                              sec_to_timecode, timecode_to_sec)

import pytest


def test_sec_to_hms() -> None:
    hms = sec_to_hms(18231.5)
    assert hms == "05:03:51.500", "default time conversion"

    hms = sec_to_hms(18231.5, use_ms=False)
    assert hms == "05:03:51", "time conversion without use_ms"

    hms = sec_to_hms(2100)
    assert hms == "00:35:00.000", "time conversion, sub-hour"

    hms = sec_to_hms(2100, use_hours=False)
    assert hms == "35:00.000", "time conversion, sub-hour, without use_hours"

    # 18231.0s -> 05:03:51.000


def test_sec_to_shortstr() -> None:
    hms = sec_to_shortstr(18231.5)
    assert hms == "5h3m51s", "time conversion with hours"

    hms = sec_to_shortstr(2102)
    assert hms == "35m2s", "time conversion sub-hour"

    hms = sec_to_shortstr(27.24601)
    assert hms == "27s", "time conversion sub-minute"


def test_hms_to_sec() -> None:
    sec = hms_to_sec("11:22:33.500")
    assert sec == 40953.5, "time conversion full"

    sec = hms_to_sec("11:22:33")
    assert sec == 40953.0, "time conversion no ms"

    sec = hms_to_sec("22:33")
    assert sec == 1353.0, "time conversion no hour"

    sec = hms_to_sec("33.5")
    assert sec == 33.5, "time conversion no hour"


def test_sec_to_timecode() -> None:
    timecode = sec_to_timecode(0.0, 60)
    assert timecode == "00:00:00:00", "time conversion with zero time"

    timecode = sec_to_timecode(3661.5, 30)
    assert timecode == "01:01:01:15", "time conversion with integer frame rate 30"

    timecode = sec_to_timecode(3661.5, 24)
    assert timecode == "01:01:01:12", "time conversion with integer frame rate 24"

    timecode = sec_to_timecode(3661.5, 25)
    assert timecode == "01:01:01:12", "time conversion with integer frame rate 25"

    with pytest.raises(ValueError) as e_info:
        timecode = sec_to_timecode(3661.5, 29.97)

        timecode = sec_to_timecode(3661.5, 30, dropframe=True)


def test_timecode_to_sec() -> None:
    sec = timecode_to_sec("00:00:00:00", 30)
    assert sec == 0.0, "time conversion with zero timecode"

    sec = timecode_to_sec("01:01:01:15", 30)
    assert sec == 3661.5, "time conversion with integer frame rate 30"

    sec = timecode_to_sec("01:01:01:12", 24)
    assert sec == 3661.5, "time conversion with integer frame rate 24"

    sec = timecode_to_sec("01:01:01:12", 25)
    assert sec == 3661.48, "time conversion with integer frame rate 25"

    with pytest.raises(ValueError) as e_info:
        sec = timecode_to_sec("01:01", 29.97)

        sec = timecode_to_sec("01:01", 30, dropframe=True)
