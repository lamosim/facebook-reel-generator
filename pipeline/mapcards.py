"""Animated route-map chapter cards (1080x1920 @ 30 fps) from a map screenshot with the route
drawn on it (onX / Gaia / AllTrails export). The red route is extracted automatically, ordered by
geodesic distance from the start, and drawn progressively: an opener that "plans" the whole loop,
DAY cards that light up the segment travelled so far, and a finish card whose arrival flare can be
timed to the song's last hit. Night/camp checkpoints get rings and small labels.

Usage:  python -m pipeline.mapcards projects/<name> [card-name ...]

Spec: work/maps.json
{
  "src": "TRIP Overview.png",                 # relative to raw/
  "start": [125, 948], "finish": [793, 1097], # route ends, source px (read them off the screenshot)
  "route_hsv": {"h_max": 12, "s_min": 120, "v_min": 120},   # optional: the route colour (default = onX red)
  "window": [30, 665, 1090, 1938],            # x, y, w, h: the 9:16 crop of the screenshot to show
  "ui_patches": [[x0, y0, x1, y1, sx, sy], ...],   # optional: cover app widgets with clean texture from (sx, sy)
  "checkpoints": {"camp1": 0.47, "camp2": 0.60},  # route fraction 0..1 (nights, portages...)
  "labels": [{"at": "camp1", "text": "NIGHT 1", "side": "right"}, {"at": "camp2", "text": "NIGHT 2", "side": "left"}],
  "cards": [
    {"name": "map1", "kind": "opener", "dur": 5.5, "display": 4.92, "big": "30 MILES", "sub": "3 DAYS  ·  2 CANOES  ·  ADIRONDACKS"},
    {"name": "map2", "kind": "day", "from": "start", "to": "camp1", "big": "DAY 2", "sub": "PONDS AND PORTAGES", "display": 2.79, "dur": 3.4},
    {"name": "map3", "kind": "day", "from": "camp1", "to": "camp2", "big": "DAY 3", "sub": "THE RIVER", "display": 2.79, "dur": 3.4},
    {"name": "map4", "kind": "finish", "from": "camp2", "dur": 5.0, "arrive": 3.2, "big": "30 MILES",
     "closing": "CARRIED IN.   PADDLED OUT.", "closing_t": 1.9, "fade_at": 3.92}
  ]
}
`display` = how long the card is on screen in the reel (its beat span); the file is rendered a bit longer.
`arrive` = seconds into the finish card when the head reaches the takeout (put it on the last hit).
Outputs work/maps/<name>.mp4 plus check frames in work/maps/checks/.
"""
from __future__ import annotations

import math
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import common
from .common import log

FPS, OW, OH = 30, 1080, 1920
FONT = common.caption_font()  # DIN Condensed Bold on macOS; bundled Barlow Condensed elsewhere


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = min(1.0, max(0.0, t))
    return 1 - (1 - t) ** 3


# ── route extraction ─────────────────────────────────────────────────────────
def extract_route(img: np.ndarray, start: tuple, hsv_spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """mask of the drawn route (the red component nearest `start`) + geodesic step distance from start."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_max, s_min, v_min = int(hsv_spec.get("h_max", 12)), int(hsv_spec.get("s_min", 120)), int(hsv_spec.get("v_min", 120))
    red = (cv2.inRange(hsv, (0, s_min, v_min), (h_max, 255, 255)) | cv2.inRange(hsv, (180 - h_max, s_min, v_min), (180, 255, 255))) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), connectivity=8)
    best, bd = None, float("inf")
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 200:
            continue
        ys, xs = np.nonzero(lab == k)
        d = float(((xs - start[0]) ** 2 + (ys - start[1]) ** 2).min())
        if d < bd:
            bd, best = d, k
    if best is None:
        common.die("mapcards: no route-coloured component found; tune route_hsv / start in work/maps.json")
    mask = lab == best
    ys, xs = np.nonzero(mask)
    i = int(np.argmin((xs - start[0]) ** 2 + (ys - start[1]) ** 2))
    sx, sy = int(xs[i]), int(ys[i])
    dist = np.full(mask.shape, -1, np.int32)
    dist[sy, sx] = 0
    q = deque([(sy, sx)])
    H, W = mask.shape
    while q:
        y, x = q.popleft()
        d = dist[y, x] + 1
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and dist[ny, nx] < 0:
                    dist[ny, nx] = d
                    q.append((ny, nx))
    mask &= dist >= 0
    return mask, dist


class MapCards:
    def __init__(self, project: common.Project, spec: dict):
        self.project, self.spec = project, spec
        self.out = project.work / "maps"
        (self.out / "checks").mkdir(parents=True, exist_ok=True)
        src = project.raw / spec["src"]
        self.img = cv2.imread(str(src))
        if self.img is None:
            common.die(f"mapcards: cannot read {src}")
        self.start, self.finish = tuple(spec["start"]), tuple(spec["finish"])
        cache_m, cache_d = self.out / "route_mask.npy", self.out / "route_dist.npy"
        if cache_m.exists() and cache_d.exists() and cache_m.stat().st_mtime > src.stat().st_mtime:
            self.route, self.dist = np.load(cache_m).astype(bool), np.load(cache_d).astype(np.float32)
        else:
            self.route, dist = extract_route(self.img, self.start, spec.get("route_hsv", {}))
            np.save(cache_m, self.route.astype(np.uint8)); np.save(cache_d, dist)
            self.dist = dist.astype(np.float32)
        self.DMAX = float(self.dist[self.route].max())
        fx, fy = self.point_at(1.0)
        if math.hypot(fx - self.finish[0], fy - self.finish[1]) > 60:
            log(f"  warning: route end ({fx:.0f},{fy:.0f}) is far from the spec finish {self.finish}; check start/finish")
        self.base = self._clean_base()
        self.thin = cv2.erode(self.route.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        self.X0, self.Y0, self.WW, self.WH = spec["window"]
        self.K = OW / self.WW
        self.center = (self.X0 + self.WW / 2, self.Y0 + self.WH / 2)
        self.check = {"start": 0.0, **{k: float(v) for k, v in spec.get("checkpoints", {}).items()}, "finish": 1.0}
        self.pt = {k: self.point_at(v) for k, v in self.check.items()}
        log(f"  route: {int(self.route.sum())} px, geodesic length {self.DMAX:.0f}; checkpoints " +
            ", ".join(f"{k}=({v[0]:.0f},{v[1]:.0f})" for k, v in self.pt.items()))

    # ── base map ──
    def _clean_base(self) -> np.ndarray:
        base = self.img.copy()
        for x0, y0, x1, y1, sx, sy in self.spec.get("ui_patches", []):
            w, h = x1 - x0, y1 - y0
            patch = base[sy:sy + h, sx:sx + w].copy()
            mask = np.full((h, w), 255, np.uint8)
            base = cv2.seamlessClone(patch, base, mask, (x0 + w // 2, y0 + h // 2), cv2.NORMAL_CLONE)
        wide = cv2.dilate(self.route.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
        base = cv2.inpaint(base, wide, 5, cv2.INPAINT_TELEA)   # remove the drawn route; we redraw it
        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= 0.85; hsv[..., 2] *= 0.82                # darker, calmer ground for labels
        base = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        cv2.imwrite(str(self.out / "checks" / "base_clean.jpg"), cv2.resize(base, (base.shape[1] // 3, base.shape[0] // 3)))
        return base

    def point_at(self, frac: float):
        d = frac * self.DMAX
        sel = self.route & (np.abs(self.dist - d) <= 4)
        ys, xs = np.nonzero(sel)
        if len(xs) == 0:
            sel = self.route & (self.dist <= d); ys, xs = np.nonzero(sel)
            i = np.argmax(self.dist[sel]); return float(xs[i]), float(ys[i])
        return float(xs.mean()), float(ys.mean())

    # ── geometry ──
    def affine(self, zoom: float, anchor):
        s = self.K * zoom
        ax, ay = anchor
        ax_o, ay_o = (ax - self.X0) * self.K, (ay - self.Y0) * self.K
        return np.array([[s, 0, ax_o - s * ax], [0, s, ay_o - s * ay]], np.float32)

    @staticmethod
    def to_out(pt, M):
        return (M[0, 0] * pt[0] + M[0, 2], M[1, 1] * pt[1] + M[1, 2])

    def compose(self, overlay_fn, M):
        canvas = self.base.copy()
        overlay_fn(canvas)
        return cv2.warpAffine(canvas, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ── layers (source space) ──
    def draw_route(self, canvas, plan_frac, trav_lo, trav_hi):
        if plan_frac > 0:
            pm = self.thin & (self.dist <= plan_frac * self.DMAX)
            canvas[pm] = (0.55 * canvas[pm] + 0.45 * np.array([235, 235, 235])).astype(np.uint8)
        if trav_hi > trav_lo:
            tm = self.route & (self.dist <= trav_hi * self.DMAX) & (self.dist >= max(0.0, trav_lo * self.DMAX - 6))
            g = cv2.GaussianBlur(cv2.dilate(tm.astype(np.uint8) * 255, np.ones((9, 9), np.uint8)), (0, 0), 9).astype(np.float32) / 255.0
            gl = g[..., None] * 0.65
            canvas[:] = np.clip(canvas * (1 - gl) + np.array([40, 120, 255], np.float32) * gl, 0, 255).astype(np.uint8)
            canvas[tm] = (30, 90, 255)
            core = cv2.erode(tm.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            canvas[core] = (90, 170, 255)

    @staticmethod
    def ring(canvas, pt, r=15, color=(230, 230, 230), thick=3, alpha=0.8, fill=False):
        x, y = int(round(pt[0])), int(round(pt[1]))
        ov = canvas.copy()
        cv2.circle(ov, (x, y), r, color, -1 if fill else thick, cv2.LINE_AA)
        cv2.addWeighted(ov, alpha, canvas, 1 - alpha, 0, canvas)

    @staticmethod
    def head(canvas, pt, t, pulse=True, flare=0.0):
        x, y = int(round(pt[0])), int(round(pt[1]))
        p = 0.5 + 0.5 * math.sin(t * 2 * math.pi * 1.1) if pulse else 0.0
        ov = canvas.copy()
        cv2.circle(ov, (x, y), int(26 + 10 * p + 80 * flare), (60, 140, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(ov, 0.45 + 0.4 * flare, canvas, 0.55 - 0.4 * flare, 0, canvas)
        cv2.circle(canvas, (x, y), 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 15, (40, 110, 255), 3, cv2.LINE_AA)

    def rings(self, canvas, keys, start_dot=True):
        for k in keys:
            self.ring(canvas, self.pt[k])
        if start_dot:
            self.ring(canvas, self.pt["start"], r=10, color=(255, 255, 255), thick=-1, alpha=0.9, fill=True)

    # ── post (output space) ──
    @staticmethod
    def gradient(out, top_a=0.88, top_h=360, bot_a=0.92, bot_h=420):
        f = out.astype(np.float32)
        ys = np.arange(OH, dtype=np.float32)
        a = np.zeros(OH, np.float32)
        a[:top_h] = top_a * (1 - ys[:top_h] / top_h) ** 1.4
        a[OH - bot_h:] = bot_a * ((ys[OH - bot_h:] - (OH - bot_h)) / bot_h) ** 1.4
        f *= (1 - a)[:, None, None]
        return np.clip(f, 0, 255).astype(np.uint8)

    _vig = None

    def vignette(self, out, strength=0.35):
        if self._vig is None:
            yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
            r = np.sqrt(((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH / 2) / (OH / 2)) ** 2)
            self._vig = np.clip(1 - strength * np.clip(r - 0.55, 0, 1.6) ** 1.5, 0, 1)[..., None]
        return np.clip(out.astype(np.float32) * self._vig, 0, 255).astype(np.uint8)

    def post(self, out):
        return self.vignette(self.gradient(out))

    @staticmethod
    def text(out, big=None, sub=None, alpha=1.0, big_y=30, sub_y=266, big_px=230, sub_px=52, cx=None):
        if alpha <= 0 or (not big and not sub):
            return out
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)

        def spaced(txt, f, y, spacing, col):
            widths = [d.textlength(ch, font=f) for ch in txt]
            total = sum(widths) + spacing * (len(txt) - 1)
            x = (OW - total) / 2 if cx is None else cx - total / 2
            for ch, w in zip(txt, widths):
                d.text((x + 3, y + 4), ch, font=f, fill=(0, 0, 0, int(160 * alpha)))
                d.text((x, y), ch, font=f, fill=col + (int(255 * alpha),))
                x += w + spacing
        if big:
            spaced(big, ImageFont.truetype(FONT, big_px), big_y, 10, (255, 255, 255))
        if sub:
            spaced(sub, ImageFont.truetype(FONT, sub_px), sub_y, 7, (255, 200, 150))
        pil = Image.alpha_composite(pil, layer).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def labels(self, out, M, alphas: dict):
        """small warm labels beside checkpoint rings (side left|right keeps clear of the right-edge UI)."""
        items = [(l, alphas.get(l["at"], 0.0)) for l in self.spec.get("labels", [])]
        items = [(l, a) for l, a in items if a > 0.002]
        if not items:
            return out
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).convert("RGBA")
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        f = ImageFont.truetype(FONT, 40)
        for l, a in items:
            x, y = self.to_out(self.pt[l["at"]], M)
            txt = l["text"]
            widths = [d.textlength(ch, font=f) for ch in txt]
            total = sum(widths) + 3 * (len(txt) - 1)
            x = x + 30 if l.get("side", "right") == "right" else x - 30 - total
            y -= 24
            for ch, w in zip(txt, widths):
                d.text((x + 2, y + 3), ch, font=f, fill=(0, 0, 0, int(170 * a)))
                d.text((x, y), ch, font=f, fill=(255, 215, 170, int(255 * a)))
                x += w + 3
        pil = Image.alpha_composite(pil, layer).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def encode(self, name, frames, n):
        path = self.out / f"{name}.mp4"
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS),
                              "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "15", "-pix_fmt", "yuv420p", "-r", str(FPS), str(path)],
                             stdin=subprocess.PIPE)
        for k, fr in enumerate(frames):
            p.stdin.write(fr.tobytes())
            if k in (0, n // 2, n - 1):
                cv2.imwrite(str(self.out / "checks" / f"{name}_f{k}.jpg"), fr)
        p.stdin.close(); p.wait()
        log(f"  wrote {path.name}")

    # ── cards ──
    def render(self, card: dict):
        kind = card["kind"]
        n = int(float(card["dur"]) * FPS)
        labeled = [l["at"] for l in self.spec.get("labels", [])]
        others = [k for k in self.check if k not in ("start",)]
        if kind == "opener":
            disp = float(card.get("display", card["dur"] - 0.5))
            def frames():
                for k in range(n):
                    t = k / FPS
                    M = self.affine(1.0 + 0.05 * ease(t / disp), self.center)
                    plan = ease_out((t - 0.35) / 2.2)
                    def ov(c):
                        self.draw_route(c, plan, 0, 0)
                        for key in others:
                            if self.check[key] <= plan + 1e-6:
                                self.ring(c, self.pt[key])
                        self.head(c, self.pt["start"], t, pulse=True)
                    out = self.post(self.compose(ov, M))
                    out = self.labels(out, M, {key: ease((plan - self.check[key]) / 0.06) for key in labeled})
                    out = self.text(out, card.get("big"), card.get("sub"), alpha=ease((t - 0.9) / 0.7))
                    fade = float(card.get("fade_in", 0.6))
                    yield (out.astype(np.float32) * min(1.0, t / fade)).astype(np.uint8) if t < fade else out
            self.encode(card["name"], frames(), n)
        elif kind == "day":
            lo, hi = self.check[card["from"]], self.check[card["to"]]
            anchor = self.pt[card.get("anchor", card["to"])]
            disp = float(card.get("display", card["dur"] - 0.6))
            draw_t0, draw_len = 0.12, max(0.9, disp - 1.0)
            def frames():
                for k in range(n):
                    t = k / FPS
                    prog = ease((t - draw_t0) / draw_len)
                    hf = lo + (hi - lo) * prog
                    M = self.affine(1.0 + 0.10 * ease(t / disp), anchor)
                    def ov(c):
                        self.draw_route(c, 1.0, 0.0, hf)
                        self.rings(c, others)
                        self.head(c, self.point_at(hf), t, pulse=(prog >= 1.0))
                    out = self.post(self.compose(ov, M))
                    al = {}
                    for key in labeled:
                        fc = self.check[key]
                        al[key] = 1.0 if fc <= lo + 1e-6 else (ease((hf - (fc - 0.02)) / 0.02) if fc <= hi + 1e-6 else 0.0)
                    out = self.labels(out, M, al)
                    out = self.text(out, card.get("big"), card.get("sub"), alpha=ease((t - 0.45) / 0.45))
                    fade = float(card.get("fade_in", 0.13))
                    yield (out.astype(np.float32) * min(1.0, t / fade)).astype(np.uint8) if t < fade else out
            self.encode(card["name"], frames(), n)
        elif kind == "finish":
            lo = self.check[card["from"]]
            arrive = float(card.get("arrive", 3.2))
            dur = float(card["dur"])
            fade_at = float(card.get("fade_at", dur - 1.0))
            def frames():
                for k in range(n):
                    t = k / FPS
                    prog = ease((t - 0.35) / (arrive - 0.35))
                    hf = lo + (1.0 - lo) * prog
                    M = self.affine(1.06 - 0.06 * ease(t / dur), self.center)
                    flare = max(0.0, 1 - abs(t - arrive) / 0.45) if t >= arrive - 0.05 else 0.0
                    def ov(c):
                        self.draw_route(c, 1.0, 0.0, hf)
                        self.rings(c, [k2 for k2 in others if k2 != "finish"])
                        if prog < 1.0:
                            self.ring(c, self.pt["finish"])
                        self.head(c, self.point_at(hf), t, pulse=(prog >= 1.0), flare=flare)
                    out = self.post(self.compose(ov, M))
                    out = self.labels(out, M, {key: 1.0 for key in labeled})
                    out = self.text(out, card.get("big"), None, alpha=ease((t - 0.3) / 0.6))
                    if card.get("closing"):
                        out = self.text(out, None, card["closing"], alpha=ease((t - float(card.get("closing_t", 1.9))) / 0.35))
                    if t > fade_at:
                        out = (out.astype(np.float32) * max(0.0, 1.0 - (t - fade_at) / 0.45)).astype(np.uint8)
                    yield out
            self.encode(card["name"], frames(), n)
        else:
            common.die(f"mapcards: unknown card kind {kind!r}")


def build_cards(project: common.Project, only: set[str] | None = None) -> None:
    spec_path = project.work / "maps.json"
    if not spec_path.exists():
        common.die(f"No {spec_path}. Write the map spec first (see pipeline/mapcards.py docstring).")
    spec = common.read_json(spec_path)
    log(f"=== MAPCARDS: {project.root.name} ===")
    mc = MapCards(project, spec)
    for card in spec["cards"]:
        if only and card["name"] not in only:
            continue
        mc.render(card)
    log(f"=== MAPCARDS complete -> {mc.out} (check frames in {mc.out / 'checks'}) ===")


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.mapcards projects/<name> [card-name ...]")
    build_cards(common.open_project(argv[0]), set(argv[1:]) or None)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
