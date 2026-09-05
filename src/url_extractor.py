# ============================================
# JOBSHIELD — JOB URL EXTRACTION
# ============================================

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


def validate_url(url):
    """Validate that the input is an HTTP/HTTPS URL."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("Job URL cannot be empty.")

    parsed = urlparse(url.strip())

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Please provide a valid HTTP or HTTPS job URL."
        )

    return url.strip()


def extract_job_text(url):
    """
    Attempt to extract readable text from a publicly
    accessible job-posting webpage.
    """

    url = validate_url(url)

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )

        response.raise_for_status()

    except requests.RequestException as error:
        raise ValueError(
            "Could not access this job URL. "
            "The website may require login, block automated requests, "
            "or be temporarily unavailable."
        ) from error

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove page elements that are usually unrelated
    # to the actual job content.
    for element in soup(
        ["script", "style", "noscript", "svg", "nav", "footer"]
    ):
        element.decompose()

    # Prefer the main page content when available.
    main = soup.find("main")

    if main:
        text = main.get_text(" ", strip=True)
    else:
        text = soup.get_text(" ", strip=True)

    # Normalize whitespace.
    text = " ".join(text.split())

        # Basic job-page validation.
    # These terms are only used to determine whether the
    # page appears to contain job-posting content.
    job_indicators = [
        "job description",
        "responsibilities",
        "requirements",
        "qualifications",
        "experience",
        "employment type",
        "apply now",
        "how to apply",
        "job title",
        "benefits",
        "salary",
        "location",
    ]

    text_lower = text.lower()

    indicator_count = sum(
        indicator in text_lower
        for indicator in job_indicators
    )

    if indicator_count < 2:
        raise ValueError(
            "This page does not appear to contain a job posting. "
            "Please provide a direct job-posting URL or paste the "
            "job description instead."
        )

    # Avoid analyzing pages with little/no readable content.
    if len(text) < 100:
        raise ValueError(
            "The page did not contain enough readable job information. "
            "Please paste the job description instead."
        )

    return text