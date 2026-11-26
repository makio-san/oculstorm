"""The storm scale explainer on the dashboard: same thresholds as the charts, sources linked, no dependencies (issue #34)."""
import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parent.parent / "docs" / "index.html").read_text(encoding="utf-8")


def _section() -> str:
    match = re.search(r'<section[^>]*id="scale".*?</section>', PAGE, re.S)
    assert match, "the storm scale section is missing"
    return match.group(0)


def test_explainer_levels_match_the_chart_thresholds():
    chart = re.search(r"const THRESHOLDS = \[(.*?)\];", PAGE).group(1)
    thresholds = [int(v) for v in re.findall(r"\[(-\d+),", chart)]
    assert thresholds == [-50, -100, -250]
    levels = [int(v) for v in re.findall(r'class="at">Dst &le; (-\d+) nT<', _section())]
    assert levels == thresholds


def test_every_level_covers_aurora_grids_and_satellites():
    section = _section()
    for topic in ("Aurora", "Power grids", "Satellites"):
        assert section.count(f"<dt>{topic}</dt>") == 3


def test_sources_are_plain_https_links_showing_their_address():
    links = re.findall(r'<a href="(https://[^"]+)">([^<]+)</a>', _section())
    assert len(links) >= 4
    assert all(href == text for href, text in links)
    hosts = {href.split("/")[2] for href, _ in links}
    assert {"www.spaceweather.gov", "doi.org", "wdc.kugi.kyoto-u.ac.jp"} <= hosts


def test_page_loads_no_external_scripts_styles_or_fonts():
    assert not re.search(r'<script[^>]+src="(?:https?:)?//', PAGE)
    assert not re.search(r'<link[^>]+href="(?:https?:)?//', PAGE)
    assert "@import" not in PAGE
    assert "@font-face" not in PAGE


def test_section_is_reachable_from_the_top_bar():
    assert '<a class="sec" href="#scale">' in PAGE
