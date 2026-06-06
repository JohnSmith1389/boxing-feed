#!/usr/bin/env python3
"""
summarize_feed.py
-----------------
Reads a source RSS feed, generates a short (2-sentence) AI summary for each item,
and writes a NEW RSS feed where each item's <description> is that summary.
Title, link, publish date, and image are preserved so dlvr.it can post
Title + Summary + Link + image to X.

Configure with environment variables (see README.md). Designed to run on a
schedule in GitHub Actions, but runs anywhere Python does.
"""

import os
import sys
import json
import re
import html
import urllib.request
from datetime import datetime, timezone

import feedparser
from feedgen.feed import FeedGenerator

# ----------------------------- config -----------------------------
SOURCE_FEED_URL = os.environ.get("SOURCE_FEED_URL", "").strip()
OUTPUT_PATH     = os.environ.get("OUTPUT_PATH", "public/feed.xml")
FEED_TITLE      = os.environ.get("FEED_TITLE", "Boxing News (AI-summarized)")
FEED_LINK       = os.environ.get("FEED_LINK", "https://example.com")
FEED_DESC       = os.environ.get("FEED_DESC", "Boxing articles with 2-sentence AI summaries.")
MAX_ITEMS       = int(os.environ.get("MAX_ITEMS", "25"))
CACHE_PATH      = os.environ.get("CACHE_PATH", "summaries_cache.json")

AI_PROVIDER     = os.environ.get("AI_PROVIDER", "gemini").lower()   # gemini | openai | anthropic | none
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_MODEL    = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_MODEL    = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

PROMPT = (
    "Summarize this boxing news article in EXACTLY 2 short sentences for a tweet. "
    "Be factual and specific (names, weight class, event, result). No hype, no hashtags, "
    "no emojis, no preamble. Keep it under 50 words total.\n\n"
    "Headline: {title}\n\nArticle text: {body}\n\nSummary:"
)

# ----------------------------- helpers -----------------------------
def strip_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def first_two_sentences(text: str) -> str:
    text = strip_html(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(parts[:2]).strip()
    return out[:280] if out else "(no summary available)"


def find_image(entry):
    # media:content
    for mc in entry.get("media_content", []) or []:
        if mc.get("url"):
            return mc["url"]
    # media:thumbnail
    for mt in entry.get("media_thumbnail", []) or []:
        if mt.get("url"):
            return mt["url"]
    # enclosure links
    for link in entry.get("links", []) or []:
        if link.get("rel") == "enclosure" and link.get("href"):
            return link.get("href")
    # <img> inside content/description
    blobs = [c.get("value", "") for c in entry.get("content", []) or []]
    blobs.append(entry.get("summary", ""))
    for blob in blobs:
        m = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', blob or "")
        if m:
            return m.group(1)
    return None


def http_post_json(url, headers, payload, timeout=40):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ai_summarize(title, body):
    body = strip_html(body)[:4000]
    prompt = PROMPT.format(title=title, body=body or title)
    try:
        if AI_PROVIDER == "gemini" and GEMINI_API_KEY:
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   + GEMINI_MODEL + ":generateContent?key=" + GEMINI_API_KEY)
            payload = {"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0.3, "maxOutputTokens": 120}}
            res = http_post_json(url, {"Content-Type": "application/json"}, payload)
            return res["candidates"][0]["content"]["parts"][0]["text"].strip()

        if AI_PROVIDER == "openai" and OPENAI_API_KEY:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": "Bearer " + OPENAI_API_KEY, "Content-Type": "application/json"}
            payload = {"model": OPENAI_MODEL, "temperature": 0.3, "max_tokens": 120,
                       "messages": [{"role": "user", "content": prompt}]}
            res = http_post_json(url, headers, payload)
            return res["choices"][0]["message"]["content"].strip()

        if AI_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
            url = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                       "Content-Type": "application/json"}
            payload = {"model": ANTHROPIC_MODEL, "max_tokens": 120,
                       "messages": [{"role": "user", "content": prompt}]}
            res = http_post_json(url, headers, payload)
            return res["content"][0]["text"].strip()
    except Exception as e:
        print("  ! AI call failed (%s); using extractive fallback" % e, file=sys.stderr)

    # Fallback: no key / provider 'none' / API error
    return first_two_sentences(body or title)


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=0)


# ----------------------------- main -----------------------------
def main():
    if not SOURCE_FEED_URL:
        sys.exit("ERROR: set SOURCE_FEED_URL")

    print("Fetching: " + SOURCE_FEED_URL)
    parsed = feedparser.parse(SOURCE_FEED_URL)
    if parsed.bozo and not parsed.entries:
        sys.exit("ERROR: could not read feed: %s" % parsed.get("bozo_exception"))
    print("  %d items found" % len(parsed.entries))

    cache = load_cache()

    fg = FeedGenerator()
    fg.load_extension("media")
    fg.title(FEED_TITLE)
    fg.link(href=FEED_LINK, rel="alternate")
    fg.description(FEED_DESC)
    fg.language("en")

    used_keys = set()
    for entry in parsed.entries[:MAX_ITEMS]:
        key = entry.get("id") or entry.get("link") or entry.get("title")
        if not key:
            continue
        used_keys.add(key)
        title = strip_html(entry.get("title", "")) or "(untitled)"
        body = ""
        if entry.get("content"):
            body = entry["content"][0].get("value", "")
        body = body or entry.get("summary", "")

        if key in cache:
            summary = cache[key]
        else:
            print("  summarizing: " + title[:60])
            summary = ai_summarize(title, body)
            cache[key] = summary

        fe = fg.add_entry()
        fe.id(key)
        fe.title(title)
        if entry.get("link"):
            fe.link(href=entry["link"])
        fe.description(summary)
        if entry.get("published_parsed"):
            dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            fe.pubDate(dt)
        img = find_image(entry)
        if img:
            fe.enclosure(img, 0, "image/jpeg")
            try:
                fe.media.content(url=img, medium="image")
            except Exception:
                pass

    cache = {k: v for k, v in cache.items() if k in used_keys}
    save_cache(cache)

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fg.rss_file(OUTPUT_PATH, pretty=True)
    print("Wrote %s with %d items" % (OUTPUT_PATH, len(used_keys)))


if __name__ == "__main__":
    main()
