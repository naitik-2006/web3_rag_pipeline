import subprocess
import os
import re

repo_path = "bitcoindev.git"
os.chdir(repo_path)

def get_commit_hashes():
    result = subprocess.run(["git", "rev-list", "--all"], capture_output=True, text=True)
    return result.stdout.strip().split("\n")

def get_commit_content(commit_hash):
    result = subprocess.run(["git", "show", commit_hash], capture_output=True, text=True)
    return result.stdout

emails = []

index = 0

for commit in get_commit_hashes():
    print(index)
    index = index + 1
    raw = get_commit_content(commit)
    # print(raw)
    # break
    try : 
        parts = raw.split("\n\n", 2)
    except Exception as e:
        print(raw)
        continue
    
    if len(parts) < 3:
        continue

    headers, subject, body = parts
    author_match = re.search(r"^\s*Author:\s*(.*)", headers, re.MULTILINE)
    date_match = re.search(r"^\s*Date:\s*(.*)", headers, re.MULTILINE)
    
    emails.append({
        "commit": commit,
        "subject": subject,
        "author": author_match.group(1).strip() if author_match else "",
        "date": date_match.group(1).strip() if date_match else "",
        "body": "Content-Type: ".join(body.strip().split("Content-Type: t")[1: ])
    })
    
    if(len(emails) == 1000): break

print(f"Extracted {len(emails)} emails")

# Save as JSONL
import json
with open("../emails_bitcoindev.jsonl", "w", encoding="utf-8") as f:
    for email in emails:
        f.write(json.dumps(email, ensure_ascii=False) + "\n")
