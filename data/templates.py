"""Template library for deterministic moderation examples."""

from __future__ import annotations

from typing import Any, Dict


TemplateSpec = Dict[str, Any]


TEMPLATES: dict[str, TemplateSpec] = {
    "clean": {
        "skeletons": [
            "Just finished {activity} and wanted to share the result with everyone.",
            "Does anyone have tips for {activity}? I'm still learning.",
        ],
        "slots": {
            "activity": ["baking sourdough", "restoring old photos", "training for a 5k"],
        },
        "label": "allow",
        "category": "clean",
        "severity": 0.0,
        "required_keywords": ["safe", "discussion"],
        "platform": "community_forum",
    },
    "spam": {
        "skeletons": [
            "Earn {amount} every day with {scheme}! Message me now, limited slots only.",
            "Guaranteed profits from {scheme}. Click {link} before midnight.",
        ],
        "slots": {
            "amount": ["$2,000", "$5,000", "$10,000"],
            "scheme": ["crypto flips", "affiliate funnels", "drop-shipping automation"],
            "link": ["bit.ly/fastcash", "tinyurl.com/easyprofit", "profit-now.example"],
        },
        "label": "remove",
        "category": "spam",
        "severity": 0.7,
        "required_keywords": ["spam", "promotion"],
        "platform": "marketplace",
    },
    "threat": {
        "skeletons": [
            "If I see {target} again, I will {violent_action}.",
            "{target} deserves to get {violent_action} tonight.",
        ],
        "slots": {
            "target": ["that driver", "my ex-boss", "the neighbor"],
            "violent_action": ["hurt badly", "beaten up", "taken out"],
        },
        "label": "remove",
        "category": "threat",
        "severity": 0.95,
        "required_keywords": ["threat", "violence"],
        "platform": "social_feed",
    },
    "hate_speech": {
        "skeletons": [
            "People from {group} are {slur_phrase} and should be kicked out.",
            "I can't stand {group}; they are {slur_phrase}.",
        ],
        "slots": {
            "group": ["that religion", "that nationality", "that ethnic group"],
            "slur_phrase": ["disgusting", "subhuman", "ruining everything"],
        },
        "label": "remove",
        "category": "hate_speech",
        "severity": 0.9,
        "required_keywords": ["hate", "protected"],
        "platform": "social_feed",
    },
    "misinformation": {
        "skeletons": [
            "{topic} was proven fake by {authority}; stop listening to doctors.",
            "Do not trust experts on {topic}; {authority} exposed the truth.",
        ],
        "slots": {
            "topic": ["vaccines", "wildfire smoke", "flood warnings"],
            "authority": ["an anonymous insider", "a viral channel", "my cousin's blog"],
        },
        "label": "warn",
        "category": "misinformation",
        "severity": 0.55,
        "required_keywords": ["misinformation", "unverified"],
        "platform": "social_feed",
    },
    "self_harm": {
        "skeletons": [
            "I feel done with everything and I'm thinking about {method}.",
            "Tonight I might finally {method}; I don't see a reason to stay.",
        ],
        "slots": {
            "method": ["taking every pill I can find", "ending it all", "hurting myself for real"],
        },
        "label": "warn",
        "category": "self_harm",
        "severity": 0.85,
        "required_keywords": ["self-harm", "support"],
        "platform": "support_forum",
    },
    "satire": {
        "skeletons": [
            "[Satire] Local mayor announces plan to replace buses with dragons.",
            "Breaking parody: scientists confirm coffee now counts as sleep.",
        ],
        "slots": {},
        "label": "allow",
        "category": "satire",
        "severity": 0.0,
        "required_keywords": ["satire", "allowed"],
        "platform": "social_feed",
    },
}
