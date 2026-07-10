"""Intentional synthetic bait for the Memory v2 privacy scanner.

Every secret/path/ID-looking value in this file must carry the exact marker
`privacy-scan: synthetic-bait-ok` so the adversarial fixture gate proves that
bait is deliberate instead of silently hidden by generic fake/example wording.
"""

SYNTHETIC_OPENAI_KEY = "sk-proj-" + ("a" * 28)  # privacy-scan: synthetic-bait-ok
SYNTHETIC_GITHUB_TOKEN = "ghp_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb-bait"  # privacy-scan: synthetic-bait-ok
SYNTHETIC_HOME_PATH = "/home/example/private/report.json"  # privacy-scan: synthetic-bait-ok
SYNTHETIC_DISCORD_ID = "1474927302512087112"  # privacy-scan: synthetic-bait-ok
