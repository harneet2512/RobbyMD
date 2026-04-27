#!/usr/bin/env python3
"""
Scrape GetTheDiagnosis.org to extract all (finding, diagnosis, sensitivity, specificity) tuples.

The site has 315 diagnoses, 1132 findings, and 1733 entries.
Each diagnosis page at /diagnosis/<Name>.htm contains HTML tables with:
  - Finding name (linked)
  - Sensitivity (%)
  - Specificity (%)
  - Comments / Study info (with PMID links)
"""

import csv
import json
import re
import sys
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser
from pathlib import Path


BASE_URL = "http://getthediagnosis.org"


class DiagnosisPageParser(HTMLParser):
    """Parse a diagnosis page to extract finding/sensitivity/specificity tuples."""

    def __init__(self, diagnosis_name: str):
        super().__init__()
        self.diagnosis = diagnosis_name
        self.entries: list[dict] = []

        # State tracking
        self.in_checker_table = False
        self.in_row = False
        self.in_cell = False
        self.cell_index = 0
        self.row_cells: list[str] = []
        self.current_cell_text = ""
        self.is_header_row = False
        self.in_a_tag = False
        self.current_section = ""

        # For tracking the section headers
        self.in_section_header = False
        self.section_text = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]):
        attr_dict = dict(attrs)

        if tag == "table" and attr_dict.get("class") == "checker_table":
            self.in_checker_table = True

        if self.in_checker_table:
            if tag == "tr":
                self.in_row = True
                self.cell_index = 0
                self.row_cells = []
                cls = attr_dict.get("class", "")
                self.is_header_row = cls == "header"

            if tag in ("td", "th") and self.in_row:
                self.in_cell = True
                self.current_cell_text = ""
                # Check for colspan which indicates "No accuracy specified"
                colspan = attr_dict.get("colspan", "1")
                if colspan == "2":
                    self.current_cell_text = "__NO_ACCURACY__"

            if tag == "a" and self.in_cell and self.cell_index == 0:
                href = attr_dict.get("href", "")
                if "/finding/" in href:
                    self.in_a_tag = True

        # Track section headers (p tags with font-size: 130%)
        if tag == "p":
            style = attr_dict.get("style", "")
            if "font-size: 130%" in style:
                self.in_section_header = True
                self.section_text = ""

    def handle_endtag(self, tag: str):
        if tag == "table" and self.in_checker_table:
            self.in_checker_table = False

        if self.in_checker_table:
            if tag in ("td", "th") and self.in_cell:
                self.in_cell = False
                self.row_cells.append(self.current_cell_text.strip())
                self.cell_index += 1

            if tag == "tr" and self.in_row:
                self.in_row = False
                if not self.is_header_row and len(self.row_cells) >= 3:
                    self._process_row()

            if tag == "a":
                self.in_a_tag = False

        if tag == "p" and self.in_section_header:
            self.in_section_header = False
            self.current_section = self.section_text.strip()

    def handle_data(self, data: str):
        if self.in_cell:
            if self.cell_index == 0 and self.in_a_tag:
                # Finding name from the <a> tag
                self.current_cell_text += data
            elif self.cell_index in (1, 2):
                # Sensitivity or specificity cells
                self.current_cell_text += data
            elif self.cell_index >= 3:
                # Comments/study
                self.current_cell_text += data

        if self.in_section_header:
            self.section_text += data

    def _process_row(self):
        finding = self.row_cells[0] if self.row_cells else ""
        sens_raw = self.row_cells[1] if len(self.row_cells) > 1 else ""
        spec_raw = self.row_cells[2] if len(self.row_cells) > 2 else ""
        comment = self.row_cells[3] if len(self.row_cells) > 3 else ""

        # Handle "No accuracy specified" case
        if "__NO_ACCURACY__" in sens_raw:
            sens = None
            spec = None
        else:
            sens = self._parse_percent(sens_raw)
            spec = self._parse_percent(spec_raw)

        # Extract PMID from comments
        pmid_match = re.search(r'PMID[:\s]*(\d+)', comment)
        pmid = pmid_match.group(1) if pmid_match else None

        # Clean up finding name
        finding = re.sub(r'\s+', ' ', finding).strip()
        # Remove "Duplicate" text
        finding = finding.replace("Duplicate", "").strip()

        if finding:
            entry = {
                "diagnosis": self.diagnosis,
                "finding": finding,
                "sensitivity": sens,
                "specificity": spec,
                "section": self.current_section,
                "comment": comment.strip()[:200] if comment else "",
                "pmid": pmid,
            }
            self.entries.append(entry)

    @staticmethod
    def _parse_percent(text: str) -> float | None:
        """Parse a percentage string like '94%' into 94.0, or return None."""
        text = text.strip().replace("%", "").strip()
        try:
            return float(text)
        except (ValueError, TypeError):
            return None


def fetch_page(url: str, retries: int = 3) -> str:
    """Fetch a page with retries and rate limiting."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (research scraper for academic use)"
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED after {retries} attempts: {e}", file=sys.stderr)
                return ""


def get_diagnosis_urls() -> list[tuple[str, str]]:
    """Get all diagnosis names and URLs from the browse page."""
    html = fetch_page(f"{BASE_URL}/browse.php?mode=dx")
    if not html:
        print("ERROR: Could not fetch browse page", file=sys.stderr)
        return []

    # Extract href="diagnosis/X.htm" patterns
    pattern = re.compile(r'href="(diagnosis/([^"]+)\.htm)"')
    results = []
    seen = set()
    for match in pattern.finditer(html):
        path = match.group(1)
        name = match.group(2).replace("_", " ")
        if path not in seen:
            seen.add(path)
            results.append((name, f"{BASE_URL}/{path}"))

    return results


def scrape_diagnosis(name: str, url: str) -> list[dict]:
    """Scrape a single diagnosis page."""
    html = fetch_page(url)
    if not html:
        return []

    parser = DiagnosisPageParser(name)
    parser.feed(html)
    return parser.entries


def main():
    output_dir = Path(__file__).parent
    csv_path = output_dir / "getthediagnosis_data.csv"
    json_path = output_dir / "getthediagnosis_data.json"

    print("Fetching diagnosis list...")
    diagnoses = get_diagnosis_urls()
    print(f"Found {len(diagnoses)} diagnoses")

    all_entries = []
    for i, (name, url) in enumerate(diagnoses):
        print(f"[{i+1}/{len(diagnoses)}] Scraping: {name}")
        entries = scrape_diagnosis(name, url)
        all_entries.extend(entries)
        # Be polite - small delay between requests
        if i < len(diagnoses) - 1:
            time.sleep(0.3)

    print(f"\nTotal entries scraped: {len(all_entries)}")

    # Count entries with actual sensitivity/specificity
    with_data = [e for e in all_entries if e["sensitivity"] is not None]
    print(f"Entries with sensitivity/specificity: {len(with_data)}")

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote JSON to {json_path}")

    # Write CSV
    fieldnames = ["diagnosis", "finding", "sensitivity", "specificity", "section", "pmid", "comment"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)
    print(f"Wrote CSV to {csv_path}")

    # Print summary stats
    unique_dx = len(set(e["diagnosis"] for e in all_entries))
    unique_fx = len(set(e["finding"] for e in all_entries))
    print(f"\nSummary: {unique_dx} diagnoses, {unique_fx} unique findings, {len(all_entries)} total entries")


if __name__ == "__main__":
    main()
