from open_storyline.utils.chronology import _parse_time, clip_time_keys, natural_key, reorder_chronologically

MEDIA = [  # load_media lists files in arbitrary (directory) order
    {"media_id": "m1", "path": "/x/大致_clip22.mp4"},
    {"media_id": "m2", "path": "/x/大致_clip5.mp4"},
    {"media_id": "m3", "path": "/x/大致_clip19.mp4"},
]
CLIPS = [
    {"clip_id": "c1", "source_ref": {"media_id": "m1", "start": 9000}},
    {"clip_id": "c2", "source_ref": {"media_id": "m1", "start": 0}},
    {"clip_id": "c3", "source_ref": {"media_id": "m2", "start": 0}},
    {"clip_id": "c4", "source_ref": {"media_id": "m3", "start": 0}},
]


def test_natural_order_when_no_metadata():
    keys = clip_time_keys(MEDIA, CLIPS, probe=lambda p: None)
    assert sorted(keys, key=keys.get) == ["c3", "c4", "c2", "c1"]  # clip5, clip19, clip22@0, clip22@9s
    assert natural_key("clip2.mp4") < natural_key("clip10.mp4")


def test_capture_time_beats_file_name_and_missing_metadata_goes_last():
    times = {"/x/大致_clip22.mp4": 100.0, "/x/大致_clip19.mp4": 200.0}
    keys = clip_time_keys(MEDIA, CLIPS, probe=times.get)
    assert sorted(keys, key=keys.get) == ["c2", "c1", "c4", "c3"]
    assert clip_time_keys(MEDIA, CLIPS, source="filename", probe=times.get)["c3"][0] == float("inf")


def test_reorder_groups_by_earliest_clip_and_renumber():
    keys = clip_time_keys(MEDIA, CLIPS, probe=lambda p: None)
    groups = [{"group_id": "group_0001", "summary": "end", "clip_ids": ["c1", "c2"]},
              {"group_id": "group_0002", "summary": "start", "clip_ids": ["c4", "c3"]}]
    out = reorder_chronologically(groups, keys)
    assert [g["summary"] for g in out] == ["start", "end"]
    assert out[0]["clip_ids"] == ["c3", "c4"] and out[1]["clip_ids"] == ["c2", "c1"]
    assert [g["group_id"] for g in out] == ["group_0001", "group_0002"]


def test_parse_quicktime_and_iso_times():
    assert _parse_time("2026-09-01T13:48:37.000000Z") == _parse_time("2026-09-01T21:48:37+08:00")
    assert _parse_time("2026-08-25T10:00:00+0900") is not None
    assert _parse_time("garbage") is None
