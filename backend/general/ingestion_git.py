"""
Git Ingestion Pipeline Module for Bitcoindev Mailing List

This module processes a cloned git repository (a mirror of the Bitcoindev mailing list),
extracts commit information (e.g. author, date, subject, body), and saves the data as JSONL.
"""

import subprocess
import os
import re
import json
import logging

# Setup logging for debugging and progress info
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define repository path relative to the current project directory.
REPO_PATH = "../bitcoindev.git"  # This is expected to be a bare-mirror or normal clone of the repo.

def get_commit_hashes() -> list:
    """
    Get a list of all commit hashes from the repository.

    Returns:
        list: A list of commit hash strings.
    """
    logger.info("Fetching commit hashes...")
    result = subprocess.run(["git", "rev-list", "--all"],
                            capture_output=True, text=True, check=True)
    commit_hashes = result.stdout.strip().split("\n")
    logger.info(f"Found {len(commit_hashes)} commits.")
    return commit_hashes

def get_commit_content(commit_hash: str) -> str:
    """
    Retrieve the full content (commit message and diff) for a given commit hash.

    Args:
        commit_hash (str): The commit hash to retrieve content for.

    Returns:
        str: The full commit content.
    """
    result = subprocess.run(["git", "show", commit_hash],
                            capture_output=True, text=True, check=True)
    return result.stdout

def parse_commit(raw: str) -> dict:
    """
    Parse the raw commit content into structured data (headers, subject, body).

    Args:
        raw (str): Raw commit content as a string.

    Returns:
        dict: A dictionary with 'subject', 'author', 'date', and 'body' keys.
    """
    # Split the commit output into header, subject, and body components.
    parts = raw.split("\n\n", 2)
    # If fewer than 3 parts, we cannot reliably extract all information.
    if len(parts) < 3:
        return {}
    
    headers, subject, body = parts

    # Extract author and date using regex.
    author_match = re.search(r"^\s*Author:\s*(.*)", headers, re.MULTILINE)
    date_match = re.search(r"^\s*Date:\s*(.*)", headers, re.MULTILINE)
    
    # Manipulate the body to remove unwanted parts.
    body_parts = body.strip().split("Content-Type: t")
    cleaned_body = "Content-Type: t".join(body_parts[1:]) if len(body_parts) > 1 else body.strip()

    return {
        "subject": subject.strip(),
        "author": author_match.group(1).strip() if author_match else "",
        "date": date_match.group(1).strip() if date_match else "",
        "body": cleaned_body
    }

def ingest_git_commits(max_emails: int = 1000) -> list:
    """
    Process the repository to extract commit details and store them as a list of dictionaries.

    Args:
        max_emails (int): Maximum number of commit messages to extract.

    Returns:
        list: A list of dictionaries, each representing a commit.
    """
    emails = []
    commit_hashes = get_commit_hashes()
    logger.info("Starting ingestion of commit data...")

    for index, commit in enumerate(commit_hashes):
        logger.info(f"Processing commit index: {index}")
        try:
            raw = get_commit_content(commit)
            parsed = parse_commit(raw)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error processing commit {commit}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error commit {commit}: {e}")
            continue
            

        if not parsed:
            continue

        parsed["commit"] = commit
        emails.append(parsed)

        if len(emails) >= max_emails:
            break

    logger.info(f"Extracted {len(emails)} emails from the repository.")
    return emails

def save_emails_as_jsonl(emails: list, output_path: str) -> None:
    """
    Save the extracted commit data as a JSONL file.

    Args:
        emails (list): List of commit dictionaries.
        output_path (str): File path to output the JSONL file.
    """
    logger.info(f"Saving emails to {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for email in emails:
            f.write(json.dumps(email, ensure_ascii=False) + "\n")
    logger.info("Data saved successfully.")

def run_git_ingestion_pipeline():
    """
    Main orchestration function for the git ingestion pipeline.
    It ensures that the repository is in the correct directory, processes commits, and saves the output.
    """
    # Change directory to the repository directory.
    os.chdir(REPO_PATH)

    emails = ingest_git_commits(max_emails=1000)
    output_file = os.path.join("..", "emails_bitcoindev.jsonl")
    save_emails_as_jsonl(emails, output_file)

if __name__ == "__main__":
    run_git_ingestion_pipeline()
