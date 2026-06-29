import os
import subprocess
import urllib.request
import json
import sys

def main():
    # 1. Get environment variables
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY environment variable is not set.")
        sys.exit(1)

    pr_number = os.environ.get("PR_NUMBER")

    # 2. Get PR diff using gh CLI
    try:
        cmd = ["gh", "pr", "diff"]
        if pr_number:
            cmd.append(pr_number)
        diff = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error getting PR diff: {e}")
        sys.exit(1)

    if not diff.strip():
        print("No diff found for this Pull Request. Skipping review.")
        return

    # 3. Construct prompt
    prompt = (
        "You are an expert software engineer performing a code review on a pull request.\n"
        "Analyze the following git diff. Identify potential bugs, security concerns, performance issues, "
        "or code style deviations. Provide constructive feedback and suggestions for improvement.\n\n"
        f"Git Diff:\n```diff\n{diff}\n```"
    )

    # 4. Call DeepSeek API using urllib
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant and a senior code reviewer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            review_text = res_json["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        sys.exit(1)

    # 5. Post review using gh CLI
    review_file = "review.md"
    with open(review_file, "w", encoding="utf-8") as f:
        f.write("### 🤖 DeepSeek AI Code Review\n\n" + review_text)

    try:
        cmd = ["gh", "pr", "review"]
        if pr_number:
            cmd.append(pr_number)
        cmd.extend(["--comment", "-F", review_file])
        subprocess.run(cmd, check=True)
        print("Review successfully posted on the PR!")
    except subprocess.CalledProcessError as e:
        print(f"Error posting review: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(review_file):
            os.remove(review_file)

if __name__ == "__main__":
    main()
