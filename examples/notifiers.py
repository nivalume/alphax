"""Telegram and Slack notifiers for ``alphax.schedule``.

Network code stays out of the library; these are examples to copy. Each is a
callable ``(subject, text)``, as ``alphax.schedule.Job`` expects::

    from notifiers import Slack, Telegram

    job = alphax.schedule.Job(lambda: alphax.config.run("daily.toml"),
                              notify=[Telegram(os.environ["TG_TOKEN"], os.environ["TG_CHAT"]),
                                      Slack(os.environ["SLACK_WEBHOOK"])])
    alphax.schedule.every(job, at="06:30")
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass


def _post(url: str, data: bytes, content_type: str, timeout: float) -> None:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"{url} answered {resp.status}")


@dataclass
class Telegram:
    """Sends through a bot: ``token`` from @BotFather, ``chat_id`` of the chat."""

    token: str
    chat_id: str
    timeout: float = 10.0

    def __call__(self, subject: str, text: str) -> None:
        body = urllib.parse.urlencode({"chat_id": self.chat_id, "text": f"{subject}\n\n{text}"[:4096]}).encode()
        _post(f"https://api.telegram.org/bot{self.token}/sendMessage", body,
              "application/x-www-form-urlencoded", self.timeout)


@dataclass
class Slack:
    """Posts to an incoming-webhook URL."""

    webhook: str
    timeout: float = 10.0

    def __call__(self, subject: str, text: str) -> None:
        body = json.dumps({"text": f"*{subject}*\n```{text}```"}).encode()
        _post(self.webhook, body, "application/json", self.timeout)
