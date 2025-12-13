import json
import dateutil
from functools import lru_cache
import os
import random
import requests
import time 
from typing import Any, Optional 
import urllib
from urllib.parse import urlparse
from bs4 import BeautifulSoup 

import boto3
import openai
from user_agent import generate_user_agent

from preprocess.retrieval.crawl.models import  NewsSearchItem, CrawledArticle
from preprocess.retrieval.config import (
    NaverAPIConfig, NaverNewsCrawlConfig,
    AWSS3Config
)
from preprocess.retrieval.config import (
    get_qgen_config, 
    get_naver_api_config, get_naver_news_crawl_config,
    get_aws_conn_config, get_aws_s3_config
)

class HTTPClient: 

    def __init__(
        self, config, user_agent=None 
    ):
        self.config = config 

        self.user_agent = user_agent 
        self.session = requests.Session()
    
    def _http_headers(self):
        headers = dict() 
        headers["User-Agent"] = self.user_agent 

        return headers
    
    def get(self, url, **kwargs):
        headers = kwargs.pop('headers', dict()) or dict() 
        http_headers = self._http_headers()
        http_headers.update(headers)

        proxies = kwargs.pop('proxies', dict()) or dict()

        resp = self.session.get(
            url, 
            headers=http_headers, 
            proxies=proxies, 
            **kwargs
        )
        resp.raise_for_status()

        return resp 
    
class NaverAPIClient(HTTPClient):

    def __init__(self, config):
        super().__init__(config)

        with open(config.llm_config.prompt_path) as read_fp:
            self.prompt_template = read_fp.read()

        self.llm_client = openai.OpenAI(
            api_key = self.config.llm_config.openai_api_key
        )

    def _naver_api_headers(self):
        headers = dict() 
        headers['X-Naver-Client-Id'] = self.config.client_id
        headers['X-Naver-Client-Secret'] = self.config.client_secret

        return headers

    def generate_query(self, data):
        prompt = self.prompt_template.replace('{ARTICLE}', data)

        if 'gpt5' in self.config.llm_config.openai_model:
            completion = self.llm_client.chat.completions.create(
                model = self.config.llm_config.openai_model, 
                messages = [
                    {'role': 'user', 'content': prompt}
                ],
                reasoning_effort = 'low',
                response_format = {'type': 'json_object'}
            )
        elif 'gpt' in self.config.llm_config.openai_model:
            completion = self.llm_client.chat.completions.create(
                model = self.config.llm_config.openai_model,
                messages = [
                    {'role': 'user', 'content': prompt}
                ],
                response_format = {'type': 'json_object'}
            )
        else:
            raise ValueError(f'{self.config.llm_config.openai_model} is not supported')

        time.sleep(0.5)

        return completion

    def generate_queries(self, prompt):
        while True:
            try:
                query = self.generate_query(prompt)
                break 
            except Exception as e: 
                print(e)
                if ('limit' in str(e)):
                    time.sleep(2)
                else: 
                    return None 
        
        return query.choices[0].message.content

    def get_queries(self, prompt):
        while True:
            try:
                raw_queries = self.generate_queries(prompt)
                if raw_queries is None: continue 
                queries = json.loads(raw_queries.strip())['queries']
                break
            except Exception as e: 
                print(e)
                print(raw_queries)
                exit(1)

        return queries
    
    def prepare_url(self, query, url_args):
        base_url, category = url_args['base_url'], url_args['category']
        
        url_args['query'] = urllib.parse.quote(query)
        main_comps = ['base_url', 'category']
        str_args = '&'.join([
            f'{k}={url_args[k]}' for k in url_args if k not in main_comps
        ])
        
        url = f"{base_url}/{category}?{str_args}"
        
        return url, url_args

    def get(self, query, start_id=1, **kwargs):
        url_args = {
            'base_url': self.config.base_url,
            'category': self.config.category,
            'display': self.config.display_num,
            'start_id': start_id
        }
        url, url_args = self.prepare_url(query, url_args)

        headers = kwargs.pop('headers', dict()) or dict() 
        api_headers = self._naver_api_headers()
        api_headers.update(headers)

        resp = super().get(
            url,
            headers=api_headers,
            **kwargs
        )

        return resp 
    
    def search_metainfos(self, data, **kwargs):
        start_id = 1 
        queries = self.get_queries(data['source'])

        metainfos = list() 
        for query in queries:
            resp = self.get(query, start_id, **kwargs)
            res_body = resp.json()

            c_metainfos = list() 
            news_items = res_body['items']
            for news_id, news in enumerate(news_items):
                c_metainfos += [
                    NewsSearchItem(
                        id=news_id, 
                        title=news['title'],
                        description=news['description'],
                        original_url=news['originallink'],
                        url=news['link'],
                        pubDate=news['pubDate']
                    )
                ]
            metainfos += c_metainfos
        
        return metainfos

class CrawlClient(HTTPClient): 

    def __init__(
        self, config,
        user_agents_num: int = 10, 
        proxy_num: int = 10, 
    ) -> None:
        super().__init__(config)

        assert (
            proxy_num < config.max_proxy_num, 
            f'Proxy 서버의 개수는 {self.config.max_proxy_num} 이하여야 합니다.' 
        ) 
        
        self.user_agents_num = user_agents_num
        self.user_agents = [
           generate_user_agent() for agent_id in range(user_agents_num)
        ]

        self._proxy_id = 0
        self.proxy_num = proxy_num
        self.proxies = random.sample(config.proxies, self.proxy_num)
    
    def _sleep(self):
        delay = random.uniform(self.config.min_delay, self.config.max_delay)
        time.sleep(delay)

    def _crawl_headers(self):     
        headers = dict()   
        headers["Referer"] = "https://www.google.com/"
        headers["Connection"] = "keep-alive"
        headers["Cache-Control"] = "max-age=0"
        headers["Upgrade-Insecure-Requests"] = "1"

        return headers

    def _pick_user_agent(self) -> str: 
        self.user_agent = random.choice(self.user_agents)
    
    def _pick_proxy(self) -> str: 
        proxy_url = self.proxies[self._proxy_id]
        self._proxy_id =  (self._proxy_id + 1) % self.proxy_num 

        return {
            'http': proxy_url, 'https': proxy_url
        }

    def _set_cookies(
        self, 
        url: str, 
        wait_until: str = "networkidle",
        wait_for_selector: Optional[str] = None, 
        timeout: int = 1500,
    ) -> str:
        with sync_playwright() as p: 
            browser = p.chromimum.launch(headless=True)
            context = browser.new_context(user_agent=self._pick_user_agent())
            page = context.new_page()

            page.goto(url, wait_until=wait_until, timeout=timeout)
            if wait_for_selector:
                page.wait_for_selector(wait_for_selector, timeout=timeout)
            
            html = page.content()

            cookies = context.cookies()
            jar = self.session.cookies 
            for c in cookies:
                jar.set(
                    c["name"], c["value"], 
                    domain = c.get("domain"),
                    path = c.get("path", "/")
                )
            
            browser.close() 
        
        return html 
    
    def get(self, url: str, **kwargs: Any) -> requests.Response: 
        self._sleep()
        self._pick_user_agent()

        headers = kwargs.pop("headers", dict()) or dict()
        crawl_headers = self._crawl_headers()
        crawl_headers.update(headers)

        proxies = self._pick_proxy()

        resp = super().get(
            url, 
            headers=crawl_headers, 
            proxies=proxies, 
            **kwargs
        )

        return resp 

class NaverNewsCrawlClient(CrawlClient):

    def _naver_headers(self):
        headers = dict()

        headers["Accept"] = (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        )
        headers['Accept-Language'] = "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
        headers['Accept-Encoding'] = "gzip, deflate, br"
        
        return headers

    def _is_naver_news(self, url: str) -> bool: 
        try:
            host = urlparse(url).hostname or "" 
        except Exception:
            return False 

        return (
            host.endswith("naver.com")
            and ("www.naver.com" in host or "n.news.naver.com" in host)
        ) 

    def _make_naver_article_id(self, url: str) -> str: 
        parsed = urlparse(url)
        parts = parsed.path.split('/')

        oid, aid = None, None 
        for i, part in enumerate(parts): 
            if (part == 'article') and (i + 2 < len(parts)): 
                oid = parts[i + 1]
                aid = parts[i + 2]
                break
        
        if oid and aid:
            return f"naver-{oid}-{aid}" 
        
        return f"naver-{parsed.netloc}{parsed.path}"
    
    def _parse_to_kst(self, dt_str): 
        dt = dateutil.parser.parse(dt_str)
        if dt.tzinfo is None: 
            dt = dt.replace(tzinfo=KST)
        else: 
            dt = dt.astimezone(KST)
        
        return dt
    
    def crawl_naver_article(self, url: str) -> dict:
        """
        네이버 뉴스 기사 1개 크롤링.
        반환: {
            "url": ...,
            "title": ...,
            "content": ...,
            "published_at": datetime | None,
        }
        """
        print("[CRAWL]: ", url)
        headers = self._naver_headers()
        resp = self.get(url, headers=headers)

        # 인코딩 보정 (가끔 깨지는 경우 대비)
        if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # ---- 1) 제목 추출 ----
        # 새 UI
        title_el = soup.select_one("h2.media_end_head_headline")
        # 구 UI / 예비
        if title_el is None:
            title_el = soup.select_one("#title_area span") or soup.select_one("h3#articleTitle")

        title = title_el.get_text(strip=True) if title_el else ""

        # ---- 2) 본문 추출 ----
        # 모바일/신 UI: #dic_area, PC: .newsct_article 등
        content_el = (
            soup.select_one("#dic_area") or
            soup.select_one("div.newsct_article") or
            soup.select_one("div#articleBodyContents") or
            soup.select_one("div#newsEndContents")
        )

        if content_el is None:
            # 그래도 못 찾으면 그냥 body 텍스트라도
            content = soup.get_text("\n", strip=True)
        else:
            # 불필요한 태그 제거 (광고, 캡션 등)
            for tag in content_el.select("script, style, span.end_photo_org, div.media_end_head_info_variety"):
                tag.decompose()
            content = content_el.get_text("\n", strip=True)

        # ---- 3) 작성 시각 추출 (가능한 경우) ----
        published_at = None

        # meta tag 쪽에서 시각 찾기
        meta_time = (
            soup.select_one("meta[property='article:published_time']") or
            soup.select_one("meta[property='og:article:published_time']")
        )
        if meta_time and meta_time.has_attr("content"):
            try:
                published_at = _parse_to_kst(meta_time["content"])
            except Exception:
                published_at = None

        # 일부 네이버 뉴스는 span.media_end_head_info_datestamp 같은 데 들어있기도 함
        if published_at is None:
            time_span = soup.select_one("span.media_end_head_info_datestamp_time") or \
                        soup.select_one("span.t11")
            if time_span:
                raw_time = time_span.get_text(strip=True)
                # 예: '2025-11-15 10:23', '2025.11.15. 오후 03:12' 등
                # 포맷이 제각각이라 dateutil로 파싱
                try:
                    published_at = _parse_to_kst(raw_time)
                except Exception:
                    published_at = None

        return CrawledArticle(
            id = self._make_naver_article_id(url),
            url = url, 
            title = title, 
            content = content, 
            published_at = published_at
        )

    def crawl_naver_articles(self, metainfos):
        error_cnt = 0 
        articles = list()
        for metainfo in metainfos: 
            if not metainfo.url: continue 
            if not self._is_naver_news(metainfo.url):
                print(f"[SKIP NON-NAVER]: {metainfo.url}")
                continue

            while len(articles) < self.config.article_num: 
                try: 
                    crawled_article = self.crawl_naver_article(metainfo.url)
                    if crawled_article.published_at is None: 
                        crawled_article.published_at = metainfo.pubDate
                    articles.append(crawled_article)
                    break
                except Exception as e:
                    error_cnt += 1 
                    if error_cnt >= self.config.err_cnt_thr: 
                        print(f"[CRAWL_ERROR] : {e} URL: {metainfo.url}")
                        break 
                    continue 
        
        return articles

@lru_cache(maxsize=1)
def get_naver_api_client(): 
    config = get_naver_api_config()
    return NaverAPIClient(config)

@lru_cache(maxsize=1)
def get_naver_news_crawl_client():
    config = get_naver_news_crawl_config()
    return NaverNewsCrawlClient(config)

@lru_cache(maxsize=1)
def get_aws_s3_client():
    config = get_aws_s3_config()

    params = { 
        'aws_access_key_id': config.conn_config.access_key_id,
        'aws_secret_access_key': config.conn_config.secret_access_key,
        'region_name': config.conn_config.default_region,
    }
    if config.endpoint_url: 
        params['endpoint_url'] = endpoint_url 

    return boto3.client('s3', **params)

def get_client(config):
    if isinstance(config, NaverAPIConfig):
        return get_naver_api_client()
    elif isinstance(config, NaverNewsCrawlConfig):
        return get_naver_news_crawl_client()
    elif isinstance(config, AWSS3Config):
        return get_aws_s3_client()
    
    raise ValueError(f'Configuraiton {config}는 지원되지 않습니다')
