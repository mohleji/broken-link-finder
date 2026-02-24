from __future__ import annotations

import argparse
import csv
import re
import sys
import threading
import tempfile
import time
import urllib.parse
import urllib.robotparser
from collections import Counter
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup


SKIP_SCHEMES = {"mailto", "tel", "sms", "javascript", "data", "file"}
DEFAULT_SKIP_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bmp",
    ".bz2",
    ".csv",
    ".doc",
    ".docx",
    ".epub",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".m4a",
    ".m4v",
    ".mid",
    ".midi",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".rss",
    ".svg",
    ".tar",
    ".tgz",
    ".tif",
    ".tiff",
    ".txt",
    ".wav",
    ".webm",
    ".webp",
    ".wmv",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


@dataclass(frozen=True)
class LinkOnPage:
    page_url: str
    link_url: str
    link_text: str


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    status_code: Optional[int]
    error: str
    final_url: str
    content_type: str
    error_html_excerpt: str
    method: str


class SessionPool:
    """
    One requests.Session per thread (requests sessions are not thread-safe).
    """

    def __init__(self, timeout_s: float, user_agent: str):
        self._timeout_s = timeout_s
        self._user_agent = user_agent
        self._local = threading.local()

    def get(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({"User-Agent": self._user_agent, "Accept": "*/*"})
            setattr(self._local, "session", s)
        return s

    @property
    def timeout(self) -> float:
        return self._timeout_s


class RateLimiter:
    """
    Enforces a minimum interval between requests to the same host (netloc),
    even when using multiple threads.
    """

    def __init__(self, min_interval_s: float):
        self._min_interval_s = max(0.0, float(min_interval_s))
        self._lock = threading.Lock()
        self._next_allowed_by_host: dict[str, float] = {}

    def wait(self, url: str) -> None:
        if self._min_interval_s <= 0:
            return
        try:
            host = urllib.parse.urlsplit(url).netloc.lower()
        except Exception:
            return

        while True:
            with self._lock:
                now = time.time()
                next_allowed = self._next_allowed_by_host.get(host, now)
                if now >= next_allowed:
                    self._next_allowed_by_host[host] = now + self._min_interval_s
                    return
                sleep_for = next_allowed - now
            time.sleep(min(sleep_for, self._min_interval_s))


def _strip_fragment_and_maybe_query(url: str, *, strip_query: bool) -> str:
    p = urllib.parse.urlsplit(url)
    query = "" if strip_query else p.query
    path = p.path or "/"
    return urllib.parse.urlunsplit((p.scheme, p.netloc, path, query, ""))


def normalize_url(url: str, *, strip_query: bool) -> str:
    url = url.strip()
    p = urllib.parse.urlsplit(url)
    if not p.scheme:
        # Keep it as-is; caller should have resolved relative URLs first.
        return url
    scheme = p.scheme.lower()
    netloc = p.netloc.lower()
    rebuilt = urllib.parse.urlunsplit((scheme, netloc, p.path, p.query, p.fragment))
    return _strip_fragment_and_maybe_query(rebuilt, strip_query=strip_query)


def is_probably_page_url(url: str, *, check_assets: bool) -> bool:
    if check_assets:
        return True
    try:
        p = urllib.parse.urlsplit(url)
    except Exception:
        return False
    path = (p.path or "").lower()
    if not path or path.endswith("/"):
        return True
    _, ext = _split_ext(path)
    if not ext:
        return True
    return ext not in DEFAULT_SKIP_EXTENSIONS


def _split_ext(path: str) -> tuple[str, str]:
    idx = path.rfind(".")
    if idx == -1:
        return path, ""
    last_slash = path.rfind("/")
    if last_slash != -1 and idx < last_slash:
        return path, ""
    return path[:idx], path[idx:]


def same_site(
    url: str,
    *,
    root_netloc: str,
    include_subdomains: bool,
) -> bool:
    try:
        p = urllib.parse.urlsplit(url)
    except Exception:
        return False
    if not p.netloc:
        return True
    netloc = p.netloc.lower()
    if netloc == root_netloc:
        return True
    if include_subdomains and netloc.endswith("." + root_netloc):
        return True
    return False


def extract_links_from_html(
    page_url: str,
    html_text: str,
    *,
    strip_query: bool,
    check_assets: bool,
    exclude_regex: Optional[re.Pattern[str]],
) -> list[LinkOnPage]:
    soup = BeautifulSoup(html_text, "html.parser")
    out: list[LinkOnPage] = []

    for a in soup.find_all("a", href=True):
        raw_href = (a.get("href") or "").strip()
        if not raw_href:
            continue
        if raw_href.startswith("#"):
            continue

        abs_url = urllib.parse.urljoin(page_url, raw_href)
        try:
            p = urllib.parse.urlsplit(abs_url)
        except Exception:
            continue

        if p.scheme and p.scheme.lower() in SKIP_SCHEMES:
            continue
        if not p.scheme or not p.netloc:
            # Skip relative/invalid after urljoin (should be absolute by now).
            continue

        abs_url = normalize_url(abs_url, strip_query=strip_query)
        if exclude_regex and exclude_regex.search(abs_url):
            continue
        if not is_probably_page_url(abs_url, check_assets=check_assets):
            continue

        text = " ".join((a.get_text(" ", strip=True) or "").split())
        out.append(LinkOnPage(page_url=page_url, link_url=abs_url, link_text=text))

    return out


def _response_excerpt(resp: requests.Response, *, limit: int) -> tuple[str, str]:
    content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if not content_type:
        content_type = ""

    excerpt = ""
    if content_type.startswith("text/") or content_type in {"application/xhtml+xml", "application/xml"}:
        # Decode small prefix to avoid huge memory usage.
        try:
            raw = resp.content[: limit * 4]
            try:
                excerpt = raw.decode(resp.encoding or "utf-8", errors="replace")
            except Exception:
                excerpt = raw.decode("utf-8", errors="replace")
        except Exception:
            excerpt = ""
        excerpt = " ".join(excerpt.split())
        excerpt = excerpt[:limit]
    return content_type, excerpt


def check_url(
    url: str,
    *,
    session_pool: SessionPool,
    excerpt_limit: int,
    rate_limiter: Optional[RateLimiter],
    retries: int,
    retry_backoff_s: float,
) -> CheckResult:
    try:
        s = session_pool.get()

        attempt = 0
        last_exc: Optional[Exception] = None
        while True:
            if rate_limiter is not None:
                rate_limiter.wait(url)

            # Prefer HEAD to save bandwidth; many servers reject it, so fall back to GET.
            method_used = "HEAD"
            resp: Optional[requests.Response] = None
            try:
                resp = s.head(url, allow_redirects=True, timeout=session_pool.timeout)
                if resp.status_code in {405, 501}:
                    raise requests.HTTPError(f"HEAD not supported ({resp.status_code})")
            except Exception:
                method_used = "GET"
                try:
                    if rate_limiter is not None:
                        rate_limiter.wait(url)
                    resp = s.get(url, allow_redirects=True, timeout=session_pool.timeout)
                except Exception as e:
                    last_exc = e
                    if attempt >= retries:
                        raise
                    time.sleep(max(0.0, retry_backoff_s) * (2**attempt))
                    attempt += 1
                    continue

            assert resp is not None
            status = int(resp.status_code)
            final_url = str(resp.url)
            content_type, excerpt = _response_excerpt(resp, limit=excerpt_limit)
            ok = 200 <= status < 400

            if ok:
                return CheckResult(
                    ok=True,
                    status_code=status,
                    error="",
                    final_url=final_url,
                    content_type=content_type,
                    error_html_excerpt="",
                    method=method_used,
                )

            # Retry on common transient / throttling statuses.
            if status in {429, 503, 502, 504} and attempt < retries:
                time.sleep(max(0.0, retry_backoff_s) * (2**attempt))
                attempt += 1
                continue

            err = f"HTTP {status}"
            return CheckResult(
                ok=False,
                status_code=status,
                error=err,
                final_url=final_url,
                content_type=content_type,
                error_html_excerpt=excerpt,
                method=method_used,
            )
    except Exception as e:
        _ = last_exc  # keep for future debugging if needed
        return CheckResult(
            ok=False,
            status_code=None,
            error=f"{type(e).__name__}: {e}",
            final_url="",
            content_type="",
            error_html_excerpt="",
            method="",
        )


def fetch_html_page(
    url: str,
    *,
    session_pool: SessionPool,
    excerpt_limit: int,
    rate_limiter: Optional[RateLimiter],
    retries: int,
    retry_backoff_s: float,
) -> tuple[Optional[str], CheckResult]:
    """
    Returns (html_text_or_none, check_result).
    If the page isn't HTML or isn't fetchable, html_text_or_none is None.
    """
    try:
        s = session_pool.get()
        attempt = 0
        while True:
            if rate_limiter is not None:
                rate_limiter.wait(url)
            resp = s.get(url, allow_redirects=True, timeout=session_pool.timeout)
            status = int(resp.status_code)
            ok = 200 <= status < 400
            content_type, excerpt = _response_excerpt(resp, limit=excerpt_limit)
            final_url = str(resp.url)

            if not ok and status in {429, 503, 502, 504} and attempt < retries:
                time.sleep(max(0.0, retry_backoff_s) * (2**attempt))
                attempt += 1
                continue
            break

        if not ok:
            return None, CheckResult(
                ok=False,
                status_code=status,
                error=f"HTTP {status}",
                final_url=final_url,
                content_type=content_type,
                error_html_excerpt=excerpt,
                method="GET",
            )

        if not (content_type.startswith("text/html") or content_type in {"application/xhtml+xml"}):
            return None, CheckResult(
                ok=True,
                status_code=status,
                error="",
                final_url=final_url,
                content_type=content_type,
                error_html_excerpt="",
                method="GET",
            )

        return resp.text, CheckResult(
            ok=True,
            status_code=status,
            error="",
            final_url=final_url,
            content_type=content_type,
            error_html_excerpt="",
            method="GET",
        )
    except Exception as e:
        return None, CheckResult(
            ok=False,
            status_code=None,
            error=f"{type(e).__name__}: {e}",
            final_url="",
            content_type="",
            error_html_excerpt="",
            method="GET",
        )


def build_robots_parser(root_url: str, user_agent: str, timeout_s: float) -> Optional[urllib.robotparser.RobotFileParser]:
    try:
        p = urllib.parse.urlsplit(root_url)
        robots_url = urllib.parse.urlunsplit((p.scheme, p.netloc, "/robots.txt", "", ""))
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        # robotparser uses urllib.request internally; use our own fetch for reliability.
        resp = requests.get(
            robots_url,
            headers={"User-Agent": user_agent, "Accept": "text/plain,*/*"},
            timeout=timeout_s,
        )
        if resp.status_code >= 400:
            return None
        rp.parse(resp.text.splitlines())
        return rp
    except Exception:
        return None


def iter_internal_pages_bfs(
    start_url: str,
    *,
    max_pages: int,
    max_depth: int,
    strip_query: bool,
    include_subdomains: bool,
    check_assets: bool,
    exclude_regex: Optional[re.Pattern[str]],
    session_pool: SessionPool,
    excerpt_limit: int,
    respect_robots: bool,
    rate_limiter: Optional[RateLimiter],
    retries: int,
    retry_backoff_s: float,
) -> Iterable[tuple[str, int, Optional[str]]]:
    """
    Yields (page_url, depth, html_text_or_none) for pages attempted.
    """
    start_url = normalize_url(start_url, strip_query=strip_query)
    root = urllib.parse.urlsplit(start_url)
    root_netloc = root.netloc.lower()

    rp = None
    if respect_robots:
        rp = build_robots_parser(start_url, session_pool.get().headers.get("User-Agent", ""), session_pool.timeout)

    q: deque[tuple[str, int]] = deque([(start_url, 0)])
    seen: set[str] = set()

    while q and len(seen) < max_pages:
        url, depth = q.popleft()
        url = normalize_url(url, strip_query=strip_query)
        if url in seen:
            continue
        if exclude_regex and exclude_regex.search(url):
            continue
        if not same_site(url, root_netloc=root_netloc, include_subdomains=include_subdomains):
            continue
        if not is_probably_page_url(url, check_assets=check_assets):
            continue
        if rp is not None:
            try:
                if not rp.can_fetch(session_pool.get().headers.get("User-Agent", "*"), url):
                    seen.add(url)
                    yield url, depth, None
                    continue
            except Exception:
                pass

        seen.add(url)
        html_text, _ = fetch_html_page(
            url,
            session_pool=session_pool,
            excerpt_limit=excerpt_limit,
            rate_limiter=rate_limiter,
            retries=retries,
            retry_backoff_s=retry_backoff_s,
        )
        yield url, depth, html_text

        if html_text is None:
            continue
        if depth >= max_depth:
            continue

        for link in extract_links_from_html(
            url,
            html_text,
            strip_query=strip_query,
            check_assets=check_assets,
            exclude_regex=exclude_regex,
        ):
            if same_site(link.link_url, root_netloc=root_netloc, include_subdomains=include_subdomains):
                q.append((link.link_url, depth + 1))


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crawl a website and report broken links as CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("start_url", help="Start URL to crawl (e.g. https://example.com/)")
    p.add_argument("-o", "--output", default="broken-links.csv", help="Output CSV path")
    p.add_argument("--max-pages", type=int, default=1000, help="Maximum number of pages to crawl")
    p.add_argument("--max-depth", type=int, default=25, help="Maximum crawl depth from start URL")
    p.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout (seconds)")
    p.add_argument("--max-workers", type=int, default=12, help="Max parallel link checks")
    p.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Minimum seconds between requests to the same host (rate limit)",
    )
    p.add_argument("--user-agent", default="broken-link-crawler/1.0", help="User-Agent header")
    p.add_argument("--check-external", action="store_true", help="Also check external links found on pages")
    p.add_argument("--include-subdomains", action="store_true", help="Treat subdomains as internal")
    p.add_argument("--strip-query", action="store_true", help="Strip querystrings when normalizing URLs")
    p.add_argument("--check-assets", action="store_true", help="Also check non-HTML links (images, PDFs, etc.)")
    p.add_argument(
        "--exclude-regex",
        default="",
        help="Skip URLs matching this regex (applied to page URLs and link URLs)",
    )
    p.add_argument("--respect-robots", action="store_true", help="Respect robots.txt disallow rules")
    p.add_argument("--excerpt-limit", type=int, default=400, help="Max chars of error HTML excerpt stored in CSV")
    p.add_argument("--retries", type=int, default=2, help="Retries for transient errors (429/502/503/504)")
    p.add_argument("--retry-backoff", type=float, default=0.6, help="Base seconds for exponential retry backoff")
    p.add_argument(
        "--report-format",
        choices=["grouped", "occurrences"],
        default="grouped",
        help="CSV layout: grouped by broken link, or one row per occurrence",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N pages (0 disables periodic progress)",
    )
    return p.parse_args(argv)


@dataclass
class BrokenAggregate:
    result: CheckResult
    pages: set[str]
    link_text_samples: list[str]
    occurrences: int = 0


def _summarize_errors(results: Iterable[CheckResult]) -> Counter[str]:
    c: Counter[str] = Counter()
    for r in results:
        if r.ok:
            continue
        if r.status_code is not None:
            c[f"HTTP {r.status_code}"] += 1
        else:
            # ExceptionType: message -> ExceptionType
            c[(r.error.split(":")[0] if r.error else "Exception")] += 1
    return c


def _write_summary_rows(
    f,
    *,
    pages_crawled: int,
    links_checked: int,
    unique_broken: int,
    broken_occurrences: int,
    error_counter: Counter[str],
) -> None:
    f.write("summary_key,summary_value\n")
    f.write(f"pages_crawled,{pages_crawled}\n")
    f.write(f"links_checked,{links_checked}\n")
    f.write(f"unique_broken_links,{unique_broken}\n")
    f.write(f"broken_link_occurrences,{broken_occurrences}\n")
    for k, v in error_counter.most_common():
        f.write(f"error_type_{k},{v}\n")
    f.write("\n")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    exclude_re = re.compile(args.exclude_regex) if args.exclude_regex else None

    session_pool = SessionPool(timeout_s=args.timeout, user_agent=args.user_agent)
    rate_limiter = RateLimiter(args.delay) if args.delay and args.delay > 0 else None
    start_url_norm = normalize_url(args.start_url, strip_query=args.strip_query)
    root_netloc = urllib.parse.urlsplit(start_url_norm).netloc.lower()

    link_cache: dict[str, CheckResult] = {}
    broken: dict[str, BrokenAggregate] = {}
    broken_unique: set[str] = set()
    broken_occurrences = 0

    if args.report_format == "occurrences":
        csv_fields = [
            "broken_link_url",
            "page_url",
            "link_text",
            "status_code",
            "error",
            "final_url",
            "content_type",
            "method",
            "error_html_excerpt",
        ]
    else:
        csv_fields = [
            "broken_link_url",
            "status_code",
            "error",
            "final_url",
            "content_type",
            "method",
            "error_html_excerpt",
            "unique_pages",
            "occurrences",
            "example_pages",
            "link_text_samples",
        ]

    pages_crawled = 0
    links_checked = 0

    body_tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            delete=False,
            prefix="broken-link-report-",
            suffix=".csv",
        ) as body_tmp:
            body_tmp_path = body_tmp.name
            w = csv.DictWriter(body_tmp, fieldnames=csv_fields)
            w.writeheader()

            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
                for page_url, depth, html_text in iter_internal_pages_bfs(
                    start_url_norm,
                    max_pages=args.max_pages,
                    max_depth=args.max_depth,
                    strip_query=args.strip_query,
                    include_subdomains=args.include_subdomains,
                    check_assets=args.check_assets,
                    exclude_regex=exclude_re,
                    session_pool=session_pool,
                    excerpt_limit=args.excerpt_limit,
                    respect_robots=args.respect_robots,
                    rate_limiter=rate_limiter,
                    retries=max(0, int(args.retries)),
                    retry_backoff_s=float(args.retry_backoff),
                ):
                    pages_crawled += 1
                    print(f"[crawling] {page_url}", file=sys.stderr)
                    if args.progress_every and args.progress_every > 0 and pages_crawled % int(args.progress_every) == 0:
                        print(
                            f"[progress] pages={pages_crawled} links_checked={links_checked} unique_broken={len(broken_unique)} now={page_url}",
                            file=sys.stderr,
                        )

                    if html_text is None:
                        continue

                    links = extract_links_from_html(
                        page_url,
                        html_text,
                        strip_query=args.strip_query,
                        check_assets=args.check_assets,
                        exclude_regex=exclude_re,
                    )
                    if not links:
                        continue

                    # Filter which links to check.
                    links_to_check: list[LinkOnPage] = []
                    for l in links:
                        internal = same_site(
                            l.link_url, root_netloc=root_netloc, include_subdomains=args.include_subdomains
                        )
                        if internal or args.check_external:
                            links_to_check.append(l)

                    if not links_to_check:
                        continue

                    # De-dupe within a page to reduce noise.
                    seen_on_page: set[str] = set()
                    links_to_check = [
                        l for l in links_to_check if not (l.link_url in seen_on_page or seen_on_page.add(l.link_url))
                    ]

                    # Check uncached links in parallel.
                    futures = {}
                    for l in links_to_check:
                        if l.link_url in link_cache:
                            continue
                        futures[
                            ex.submit(
                                check_url,
                                l.link_url,
                                session_pool=session_pool,
                                excerpt_limit=args.excerpt_limit,
                                rate_limiter=rate_limiter,
                                retries=max(0, int(args.retries)),
                                retry_backoff_s=float(args.retry_backoff),
                            )
                        ] = l.link_url

                    for fut in as_completed(futures):
                        url = futures[fut]
                        link_cache[url] = fut.result()

                    # Write broken links for this page.
                    for l in links_to_check:
                        res = link_cache.get(l.link_url)
                        if res is None:
                            continue
                        links_checked += 1
                        if res.ok:
                            continue
                        broken_occurrences += 1
                        broken_unique.add(l.link_url)

                        if args.report_format == "occurrences":
                            w.writerow(
                                {
                                    "broken_link_url": l.link_url,
                                    "page_url": l.page_url,
                                    "link_text": l.link_text,
                                    "status_code": "" if res.status_code is None else res.status_code,
                                    "error": res.error,
                                    "final_url": res.final_url,
                                    "content_type": res.content_type,
                                    "method": res.method,
                                    "error_html_excerpt": res.error_html_excerpt,
                                }
                            )
                        else:
                            agg = broken.get(l.link_url)
                            if agg is None:
                                agg = BrokenAggregate(result=res, pages=set(), link_text_samples=[], occurrences=0)
                                broken[l.link_url] = agg
                            agg.pages.add(l.page_url)
                            agg.occurrences += 1
                            if (
                                l.link_text
                                and l.link_text not in agg.link_text_samples
                                and len(agg.link_text_samples) < 5
                            ):
                                agg.link_text_samples.append(l.link_text)

                if args.report_format == "grouped":
                    # Append aggregates now in a stable order.
                    for broken_url in sorted(broken.keys()):
                        agg = broken[broken_url]
                        example_pages = sorted(agg.pages)[:20]
                        w.writerow(
                            {
                                "broken_link_url": broken_url,
                                "status_code": "" if agg.result.status_code is None else agg.result.status_code,
                                "error": agg.result.error,
                                "final_url": agg.result.final_url,
                                "content_type": agg.result.content_type,
                                "method": agg.result.method,
                                "error_html_excerpt": agg.result.error_html_excerpt,
                                "unique_pages": len(agg.pages),
                                "occurrences": agg.occurrences,
                                "example_pages": " | ".join(example_pages),
                                "link_text_samples": " | ".join(agg.link_text_samples),
                            }
                        )

        # Write final output: summary + body.
        error_counter = (
            _summarize_errors(broken[a].result for a in broken)
            if args.report_format == "grouped"
            else _summarize_errors(r for r in link_cache.values() if not r.ok)
        )
        with open(args.output, "w", encoding="utf-8", newline="") as out:
            _write_summary_rows(
                out,
                pages_crawled=pages_crawled,
                links_checked=links_checked,
                unique_broken=len(broken_unique),
                broken_occurrences=broken_occurrences,
                error_counter=error_counter,
            )
            with open(body_tmp_path, "r", encoding="utf-8") as body_in:
                for chunk in iter(lambda: body_in.read(1024 * 1024), ""):
                    out.write(chunk)
    finally:
        if body_tmp_path:
            try:
                import os

                os.unlink(body_tmp_path)
            except Exception:
                pass

    print(
        f"Done. Crawled {pages_crawled} page(s), checked {links_checked} link(s). "
        f"Broken occurrences: {broken_occurrences}. Unique broken: {len(broken_unique)}. "
        f"CSV: {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

