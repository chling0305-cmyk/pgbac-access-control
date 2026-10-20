# -*- coding: utf-8 -*-
"""SVG -> PNG 渲染（Chrome headless）。

Windows 上没有 cairosvg / inkscape / rsvg，用本机 Chrome 的无头截图：
把 SVG 内联进一个零边距 HTML，窗口尺寸设成目标像素，用 --force-device-scale-factor=1
保证 1 CSS px = 1 PNG px。字体走系统 Arial，与 SVG 里声明的 font-family 一致。

Chrome 路径可用环境变量 `CHROME_EXE` 覆盖；临时 HTML 与 profile 落在系统临时目录，
不写进 _tools（否则每次渲染都在版本库里留下生成物）。
"""
import os
import re
import subprocess
import sys
import tempfile

CHROME = os.environ.get("CHROME_EXE") or r"C:\Program Files\Google\Chrome\Application\chrome.exe"
WORK = os.path.join(tempfile.gettempdir(), "pgbac_svg_render")


def render(svg_text, width_px, height_px, out_png, chrome=None):
    """把 svg_text 按 width_px x height_px 渲染成 PNG。"""
    chrome = chrome or CHROME
    out_png = os.path.abspath(out_png)          # chrome 不认相对路径
    os.makedirs(WORK, exist_ok=True)
    svg = re.sub(r'<svg\b([^>]*)>', _fix_size(width_px, height_px), svg_text, count=1)
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        "<style>html,body{margin:0;padding:0;background:#fff;}"
        "svg{display:block}</style></head><body>"
        + svg
        + "</body></html>"
    )
    fd, html_path = tempfile.mkstemp(suffix=".html", dir=WORK)
    os.close(fd)
    with open(html_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    prof = os.path.join(WORK, "chrome-profile")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--no-first-run",
        "--no-default-browser-check",
        "--force-device-scale-factor=1",
        "--user-data-dir=" + prof,
        "--window-size=%d,%d" % (width_px, height_px),
        "--screenshot=" + out_png,
        "file:///" + html_path.replace("\\", "/"),
    ]
    p = subprocess.run(cmd, capture_output=True, timeout=180)
    os.unlink(html_path)
    if not os.path.exists(out_png) or os.path.getsize(out_png) < 2000:
        raise RuntimeError("渲染失败：%s\n%s\n%s"
                           % (out_png, p.stderr.decode("utf-8", "ignore")[-1200:],
                              p.stdout.decode("utf-8", "ignore")[-600:]))
    return out_png


def _fix_size(w, h):
    def sub(m):
        attrs = m.group(1)
        attrs = re.sub(r'\swidth="[^"]*"', "", attrs)
        attrs = re.sub(r'\sheight="[^"]*"', "", attrs)
        return '<svg width="%d" height="%d"%s>' % (w, h, attrs)
    return sub


def png_size(path):
    import struct
    b = open(path, "rb").read(32)
    assert b[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", b[16:24])


if __name__ == "__main__":
    src, out, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    t = open(src, encoding="utf-8").read()
    render(t, w, h, out)
    print(out, png_size(out), os.path.getsize(out) // 1024, "KB")
