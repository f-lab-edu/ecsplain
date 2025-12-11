from dataclasses import dataclass, asdict
from datetime import datetime 
import json
from typing import Optional 

@dataclass
class NewsSearchItem:
    id: str
    title: str 
    description: str 
    url: str 
    original_url: str 
    pubDate: str

@dataclass
class CrawledArticle:
    id: str 
    title: str 
    url: str 
    content: str 
    published_at: Optional[datetime]

    def to_dict(self):
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)
