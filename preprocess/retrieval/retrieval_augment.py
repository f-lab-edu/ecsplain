import argparse
from bs4 import BeautifulSoup 
import dateutil
import json 
import os 
import openai
import requests
import sys 
import time 
import urllib
from urllib.parse import urlparse


ERR_CNT_THR=10
DISPLAY_NUM=100

CLIENT_ID=os.environ['NAVER_API_CLIENT_ID']
CLIENT_SECRET=os.environ['NAVER_API_CLIENT_SECRET']

BASE_URL = "https://openapi.naver.com/v1/search"

KST = dateutil.tz.gettz("Asia/Seoul")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0 Safari/537.36"
    )
}

def read_directory_files(config):
    data = list()
    file_names = os.listdir(config.read_directory_path)
    for file_name in file_names: 
        file_path = os.path.join(
            config.read_directory_path, file_name
        )
        with open(file_path, 'r', encoding='utf-8') as read_fp: 
            for line in read_fp.readlines():
                js = json.loads(line)
                data.append(js)
    
    return data 


def get_args(): 
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--read_directory_path', type = str, default = './stage_3_filtered_newses' 
    )
    argparser.add_argument(
        '--write_directory_path', type = str, default = './stage_4_retrieval_augment'
    )

    argparser.add_argument(
        '--prompt_fp', type=str, default='./prompt_template/retrieval_augment.txt'
    )
    argparser.add_argument(
        '--qgen_model', type=str, default='gpt-4o'
    )

    argparser.add_argument(
        '--article_num', type=int, default=10
    )

    argparser.add_argument(
        '--test_sample_num', type=int, default=0
    )

    return argparser.parse_args()

def generate_query(config, llm_client, prompt):
    if 'gpt5' in config.qgen_model:
        completion = llm_client.chat.completions.create(
            model = config.qgen_model, 
            messages = [
                {'role': 'user', 'content': prompt}
            ],
            reasoning_effort = 'low',
            response_format = {'type': 'json_object'}
        )
    elif 'gpt' in config.qgen_model:
        completion = llm_client.chat.completions.create(
            model = config.qgen_model,
            messages = [
                {'role': 'user', 'content': prompt}
            ],
            response_format = {'type': 'json_object'}
        )
    else:
        raise ValueError(f'{config.qgen_model} is not supported')

    time.sleep(0.5)

    return completion


def generate_queries(config, llm_client, prompt):
    while True:
        try:
            query = generate_query(config, llm_client, prompt)
            break 
        except Exception as e: 
            print(e)
            if ('limit' in str(e)):
                time.sleep(2)
            else: 
                return None 
    
    return query.choices[0].message.content

def get_queries(config, llm_client, prompt):
    while True:
        try:
            raw_queries = generate_queries(config, llm_client, prompt)
            if raw_queries is None: continue 
            queries = json.loads(raw_queries.strip())['queries']
            break
        except Exception as e: 
            print(e)
            print(raw_queries)
            exit(1)

    return queries

def prepare_url(base_url, category, url_args=None, restful_type='get'):
    if restful_type == 'get':
        url_args['query'] = urllib.parse.quote(url_args['query'])
        str_args = '&'.join([f'{k}={url_args[k]}' for k in url_args])
        
        url = f"{base_url}/{category}?{str_args}"
    else: 
        raise ValueError(f'{restful_type} is not supported')
    
    return url, url_args

def construct_request(url, data=None):
    if data is None:
        request = urllib.request.Request(url)
        
        request.add_header('X-Naver-Client-Id', CLIENT_ID)
        request.add_header('X-Naver-Client-Secret', CLIENT_SECRET)
    else: 
        request = urllib.request.Request(url, data=data)

        request.add_header('X-Naver-Client-Id', CLIENT_ID)
        request.add_header('X-Naver-Client-Secret', CLIENT_SECRET)
        request.add_header("Content-Type", "application/x-www-form-urlencoded")

    return request

def get_response(base_url, category, args): 
    url, _ = prepare_url(base_url, category, args, restful_type='get')

    request = construct_request(url)
    response = urllib.request.urlopen(request)

    args.update({
        'url': url, 'data': None 
    })

    return response 


def is_naver_news(url: str) -> bool: 
    try:
        host = urlparse(url).hostname or "" 
    except Exception:
        return False 

    return (
        host.endswith("naver.com")
        and ("www.naver.com" in host or "n.news.naver.com" in host)
    ) 

def make_naver_article_id(url: str) -> str: 
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

def parse_to_kst(dt_str): 
    dt = dateutil.parser.parse(dt_str)
    if dt.tzinfo is None: 
        dt = dt.replace(tzinfo=KST)
    else: 
        dt = dt.astimezone(KST)
    
    return dt

def crawl_naver_article(url: str) -> dict:
    """
    네이버 뉴스 기사 1개 크롤링.
    반환: {
        "url": ...,
        "title": ...,
        "content": ...,
        "published_at": datetime | None,
    }
    """
    print("[crawl] url =", url)
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()

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
            published_at = parse_to_kst(meta_time["content"])
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
                published_at = parse_to_kst(raw_time)
            except Exception:
                published_at = None

    return {
        "id": make_naver_article_id(url),
        "url": url,
        "title": title,
        "content": content,
        "published_at": published_at,
    }

def crawl_naver_articles(config, res_body):
    articles = list() 
    for article_id, article in enumerate(res_body['items']): 
        if len(articles) >= config.article_num: break
        try:
            if not article['link']: continue 
            if not is_naver_news(article['link']): 
                print("[SKIP NON-NAVER]", article['link'])
                continue
            crawled = crawl_naver_article(article['link'])
            article.update({
                'id': crawled['id'], 'content': crawled
            })
            articles.append(article)
        except Exception as e: 
            print("crawl error:", e, "url:", article.get('link'))
    
    return articles


def crawl(config, url_args):
    articles = list() 
    start_id, error_cnt = 1, 0  
    while (start_id < 1000) and (len(articles) < config.article_num):
        try:
            url_args['start'] = start_id
            response = get_response(BASE_URL, 'news', url_args)
            res_code = response.getcode() 

            if res_code == 200: 
                res_body = json.loads(response.read().decode())
                articles += crawl_naver_articles(config, article)

                start_id = start_id + DISPLAY_NUM
                error_cnt = 0 
            else: 
                print(f'ERROR CODE: {res_code}')
        except Exception as e: 
                print(e)
                error_cnt += 1 
                if error_cnt >= ERR_CNT_THR: break 
                continue 
    
    return articles


def main(config):
    data = read_directory_files(config)

    prompt_template = open(config.prompt_fp).read()
    llm_client = openai.OpenAI(
        api_key = os.environ.get('OPENAI_API_KEY')
    )

    for i, js in enumerate(data): 
        threshold_date = parse_to_kst(js['metadata']['date'])

        prompt = prompt_template.replace('{ARTICLE}', js['source'])
        queries = get_queries(config, llm_client, prompt) 
        js['retrievals'] = list() 
        for query in queries: 
            url_args = {
                'query': query, 'display': DISPLAY_NUM, 
            }
            articles = crawl(config, url_args)
            if len(articles) > 0:
                js['retrievals'].extend([ article['id'] for article in articles])
        
        os.makedirs(config.write_directory_path, exist_ok=True)
        eval_path = os.path.join(config.write_directory_path, 'retrieval_augmented_newses.jsonl')
        with open(eval_path, 'a', encoding='utf-8') as eval_fp: 
            eval_fp.write(json.dumps(js, ensure_ascii=False) + '\n')

        pool_path = os.path.join(config.write_directory_path, 'retrieval_pool.jsonl')
        with open(pool_path, 'a', encoding='utf-8') as pool_fp: 
            to_write = '\n'.join([
                json.dumps(article, ensure_ascii=False) for article in articles
            ])
            pool_fp.write(to_write + '\n')
    


if __name__ == '__main__':
    config = get_args() 
    main(config)

