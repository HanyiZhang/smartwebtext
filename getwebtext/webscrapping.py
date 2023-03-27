from keyword_search_initializer import prnews_kw_company_finder
from string_utils import *
import requests
from bs4 import BeautifulSoup
import lxml.html
import re
import time
from dateutil.parser import parse


kw_company = prnews_kw_company_finder()
if not kw_company.company_uris:
    kw_company.run_search()


def extract_date(body):
    dateparts = keep_alphanum(body.split("/PRNewswire/")[0])
    datestr_back = ' '.join(dateparts.split()[-3:])
    datestr_front = ' '.join(dateparts.replace('Share this article', '').split()[:3])
    datestr = ' '
    try:
        if hasNumbers(datestr_back):
            dateform = parse(datestr_back, fuzzy_with_tokens=True)[0]
            datestr = str(dateform).split()[0]
        elif hasNumbers(datestr_front):
            dateform = parse(datestr_front, fuzzy_with_tokens=True)[0]
            datestr = str(dateform).split()[0]
    except:
        pass
    return datestr


def containkwOR(sentence, keywords):
    for word in keywords:
        if word in sentence:
            return True
    return False


def containkwAND(sentence, keywords):
    return all([w in sentence for w in keywords])


def findLocalWords(str1, keywords_rule,center='analytics', nlocal=12):
    words_list = clean_str(str1.lower()).split(center)
    if containkwOR(' '.join(words_list[0].split()[-(nlocal + 1):]), keywords_rule):
        return True
    if containkwOR(' '.join(words_list[-1].split()[:nlocal]), keywords_rule):
        return True
    for words in words_list[1:-1]:
        if containkwOR(' '.join(words.split()[:nlocal]), keywords_rule):
            return True
        if containkwOR(' '.join(words.split()[-(nlocal + 1):]), keywords_rule):
            return True
    return False


class websearch():

    stock_market = ['NYSE', 'NASDAQ', 'Nasdaq']
    reject_titleWords = ['market', 'award', 'report', 'industry', 'universit', 'hospital', 'survey', 'forum']
    #reject_titleWords = ['market', 'report', 'universit', 'hospital', 'survey', 'forum']

    def __init__(self, output_file, min_len=10):
        self.output_file = open(output_file, 'w', encoding='utf-8')
        self.output_file.write("Number\tCompany\tStock\tdate\tTitle_of_Press_Announce\tBody\tLink:\n")
        self.min_len = min_len
        self.num_records = 0
        self.titles = []

    def _containRule(self, sentence, center_word):
        if center_word not in sentence:
            return False
        # return containkwOR(sentence.lower(),keywords_rule)
        return findLocalWords(sentence, center_word)

    def scrap_rule0(self, uri,company_name, login=None, loginform=None):
        soup = self.get_soup(uri, login, loginform)
        if soup is None:
            return
        # content - Body
        # paragraph - Related Paragraphs
        # title - Title of Press Announce
        # uri - link

        title = '-'.join(uri.split('/')[-1].split('-')[:-1])
        if containkwOR(title.lower(), self.reject_titleWords):
            return

        if title in self.titles:
            return
        else:
            self.titles.append(title)

        sid = soup.text.find('Share this article')
        eid = soup.text.find('\n×\nModal title')
        content = soup.text[sid:eid].replace('Share this article', '')
        content = re.sub(r'\n\s*\n', ' ', content)

        if not containkwOR(content, self.stock_market):
            return

        stocks = find_stock_code(content)
        # if containkwOR( content,or_contain) and len(stocks)>0:
        if len(stocks) > 0:

            content=content.replace('\n',';;;;')
            content = content.replace('\t', ' ')
            datestr = extract_date(content)

            sid=content.find("/PRNewswire/ --")
            sid+=len("/PRNewswire/ --")
            content= content[sid:]
            self.output_file.write("{:d}\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\n" \
                                   .format(self.num_records + 1, company_name, \
                                           ','.join(stocks), datestr, title, content, uri))
            self.num_records += 1

        return soup

    def get_soup(self, uri, login=None, loginform=None):
        ntries = 10
        for i in range(ntries):
            try:
                if not login:
                    html = requests.get(uri).text
                else:
                    html = login(uri, loginform)
                soup = BeautifulSoup(html, 'html.parser')
                return soup
            except:
                time.sleep(1000)
                continue
        return


def scrap_companies(scrapper, company_list):
    nvalid_uri = 0
    for icc, company_link in enumerate(company_list):
        print('{:d}/{:d}: {:s}'.format(icc, len(company_list), company_link))
        # main_uri = "https://www.prnewswire.com/search/news/?keyword=analytics&pagesize=25&page="+str(ppno)
        company = company_link[:-1].split('/')[-1]
        ii = 1
        while True:
            company_uri = company_link + '?page=' + str(ii) + '&pagesize=25'
            print(company,"-", company_uri)

            main_soup = scrapper.get_soup(company_uri)
            if main_soup is None:
                continue

            found_scrap = False
            for link in main_soup.find_all('a'):
                uri = link.get('href')
                if not uri:
                    continue
                if 'news-releases' in uri and ('.html' in uri):
                    print(uri)
                    nvalid_uri += 1
                    soup = scrapper.scrap_rule0(kw_company.link_uri + uri, company)
                    if soup:
                        found_scrap = True

            if not found_scrap:
                break
            ii += 1

    print('{} records found.'.format(scrapper.num_records))
    print('{} valid uri.'.format(nvalid_uri))
    return scrapper


def scrap_search(scrapper, search_count):
    nvalid_uri = 0
    n_search = len(search_count.keys())
    for it, term in enumerate(search_count.keys()):
        print('{}/{}'.format(it + 1, n_search))
        for ppno in range(1, search_count[term] // 25 + 2):
            main_uri = "https://www.prnewswire.com/search/news/?keyword=" + term + "&pagesize=25&page=" + str(ppno)
            print(main_uri)

            main_soup = scrapper.get_soup(main_uri)
            for link in main_soup.find_all('a'):
                uri = link.get('href')
                if not uri:
                    continue
                if 'news-releases' in uri and ('.html' in uri):
                    print(uri)
                    soup = scrapper.scrap_rule0(kw_company.link_uri + uri)
                    nvalid_uri += 1
    print('{} records found.'.format(scrapper.num_records))
    print('{} valid uri.'.format(nvalid_uri))
    return scrapper


def main():
    scrapper = websearch('data_model/extracted_text.txt')
    scrapper = scrap_companies(scrapper, kw_company.company_uris)
    # scrapper=scrap_search(scrapper, {'analytics':9564})

    # search for analytics+all providers: Recall is too low
    # scrap_search(scrapper,{**challengers, **providers})
    return


if __name__ == "__main__":
    main()