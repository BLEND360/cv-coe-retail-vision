import importlib

main = importlib.import_module("main")


def test_retail_brands_share_one_class_key():
    blend = main._class_key(main.classes_for_brand("blend360"))
    ua = main._class_key(main.classes_for_brand("under-armour"))
    hyatt = main._class_key(main.classes_for_brand("hyatt"))
    assert blend == ua          # retail brands collapse to one model
    assert blend != hyatt       # hospitality is a distinct model
    assert len(main.classes_for_brand("hyatt")) == 9
    assert len(main.classes_for_brand("blend360")) == 14


def test_unknown_brand_falls_back_to_retail():
    assert main.classes_for_brand("does-not-exist") == main.RETAIL_CLASSES


def test_video_path_for_brand():
    assert main.video_path_for_brand("hyatt").endswith("Hyatt.mp4")
    assert main.video_path_for_brand("blend360").endswith("The BLEND360 Approach.mp4")
    assert main.video_path_for_brand("under-armour").endswith("Under-Armour.mp4")
    # Unknown brand falls back to a real path string (env/default), not None
    assert isinstance(main.video_path_for_brand("nope"), str)
