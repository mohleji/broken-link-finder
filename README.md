## Broken link website crawler (Python)

This is a small crawler that:
- Starts from a URL
- Traverses internal pages (BFS)
- Checks links found on each page
- Writes a CSV report of **broken links**, including HTTP status code/exception and an **error HTML excerpt**

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Internal links only (recommended default):

```bash
python broken_link_crawler.py "https://example.com" --output report.csv
```

By default, the crawler rate-limits requests to the same host with `--delay 0.2` (helps avoid transient 503/429).

Also check external links:

```bash
python broken_link_crawler.py "https://example.com" --check-external --output report.csv
```

Limit crawl size and be polite:

```bash
python broken_link_crawler.py "https://example.com" \
  --max-pages 500 \
  --max-depth 10 \
  --max-workers 8 \
  --delay 0.5 \
  --respect-robots \
  --output report.csv
```

Exclude URLs (regex):

```bash
python broken_link_crawler.py "https://example.com" \
  --exclude-regex "/(logout|cart|wp-admin)/" \
  --output report.csv
```

## CSV columns

The output CSV begins with a small **summary section** (key/value rows), then a blank line, then the main report table.

### Default report format: `grouped`

This is the default format (`--report-format grouped`). One row per **unique broken link**, with pages listed:

- **broken_link_url**: the broken link URL
- **status_code**: HTTP status (blank for exceptions)
- **error**: `HTTP <code>` or `ExceptionType: message`
- **final_url**: final URL after redirects (if any)
- **content_type**: response `Content-Type` (if available)
- **method**: `HEAD` or `GET` used for the check
- **error_html_excerpt**: first N chars of error response body (HTML/text only)
- **unique_pages**: how many distinct pages contained this broken link
- **occurrences**: total times the link appeared across pages (after per-page de-dupe)
- **example_pages**: up to 20 example pages where it was found
- **link_text_samples**: up to 5 example anchor texts

### Alternate report format: `occurrences`

Use `--report-format occurrences` for one row per broken-link occurrence (broken link first, then the page it was found on):

- **broken_link_url**: the broken link URL
- **page_url**: the page where the link was found
- **link_text**: anchor text (best-effort)
- **status_code**: HTTP status (blank for exceptions)
- **error**: `HTTP <code>` or `ExceptionType: message`
- **final_url**: final URL after redirects (if any)
- **content_type**: response `Content-Type` (if available)
- **method**: `HEAD` or `GET` used for the check
- **error_html_excerpt**: first N chars of error response body (HTML/text only)

## Notes / tips

- By default the crawler avoids checking obvious non-page assets (PDFs, images, etc.). Use `--check-assets` if you want those checked too.
- If your site has many URL variants via query params, consider `--strip-query` to reduce duplicate checks.
- If you see flaky `HTTP 503`/`HTTP 429`, increase `--delay` and/or lower `--max-workers`. The crawler also retries transient statuses by default (`--retries 2`).

