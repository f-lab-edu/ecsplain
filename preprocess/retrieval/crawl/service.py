from typing import Dict, List, Tuple

from preprocess.retrieval.crawl.clients import get_client


class NaverNewsService:
    def __init__(
        self, 
        config,
    ):
        self.config = config 

        self.api_client = get_client(self.config.api_config)  
        self.crawl_client = get_client(self.config.crawl_config)

    
    def fetch_news(self, data) -> Tuple[Dict, List]:
        metainfos = self.api_client.search_metainfos(data)
        newses = self.crawl_client.crawl_naver_articles(metainfos)
        data['retrievals'] = [ news.id for news in newses ]
        
        return data, newses


