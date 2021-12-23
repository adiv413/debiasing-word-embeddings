"""
Creates a corpus from Wikipedia dump file.

Inspired by:
https://github.com/panyang/Wikipedia_Word2vec/blob/master/v1/process_wiki.py

Link to the wikipedia dump: https://dumps.wikimedia.org/enwiki/latest/
Filename for download: enwiki-latest-pages-articles.xml.bz2
"""

import sys
from gensim.corpora.wikicorpus import *
from alive_progress import alive_bar
import traceback

# IF BAD SYMBOLS, OVERRIDE THIS METHOD TO ACCOUNT FOR THAT

def tokenize(content):
    #override original method in wikicorpus.py
    return [token.encode('utf8') for token in content.split() 
           if len(token) <= 15 and not token.startswith('_') 
           and not token.startswith('=') 
           and not token.startswith('*') 
           and not token.endswith('=')]

def process_article(args):
   # override original method in wikicorpus.py
    text, lemmatize, title, pageid = args
    text = filter_wiki(text)
    if lemmatize:
        result = utils.lemmatize(text)
    else:
        result = tokenize(text)
    return result, title, pageid


class MyWikiCorpus(WikiCorpus): 
    # have a dummy value in the dictionary field so that dictionary creation isnt started, which takes a VERY long time
    def __init__(self, fname, processes=None, lemmatize=utils.has_pattern(), dictionary={'x' : 1}, filter_namespaces=('0',)):
        WikiCorpus.__init__(self, fname, processes, lemmatize, dictionary, filter_namespaces)

    def get_texts(self):
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0
        texts = ((text, self.lemmatize, title, pageid) for title, text, pageid in extract_pages(bz2.BZ2File(self.fname), self.filter_namespaces))
        pool = multiprocessing.Pool(self.processes)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(texts, chunksize=10 * self.processes, maxsize=1):
            for tokens, title, pageid in pool.imap(process_article, group):  # chunksize=10):
                articles_all += 1
                positions_all += len(tokens)
                # article redirects and short stubs are pruned here
                if len(tokens) < ARTICLE_MIN_WORDS or any(title.startswith(ignore + ':') for ignore in IGNORED_NAMESPACES):
                    continue
                articles += 1
                positions += len(tokens)
                if self.metadata:
                    yield (tokens, (pageid, title))
                else:
                    yield tokens
        pool.terminate()

        logger.info(
            "finished iterating over Wikipedia corpus of %i documents with %i positions"
            " (total %i articles, %i positions before pruning articles shorter than %i words)",
            articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS)
        self.length = articles  # cache corpus length

def make_corpus(out_f):
    output = open(out_f, 'w')
    infiles = [
            "enwiki-latest-pages-articles.xml.bz2"
        ]

    for in_f in infiles: 
        print("Processing file:", in_f, '\n', flush=True)   
        wiki = MyWikiCorpus(in_f)
        print("\nLoaded text", flush=True)

        with alive_bar(int(5350432)) as bar:
            for text in wiki.get_texts():
                try:
                    output.write(bytes(' '.join([i.decode('utf-8') for i in text]), 'utf-8').decode('utf-8') + '\n')
                except Exception as e:
                    # traceback.print_exc()
                    continue
                bar()
    output.close()
    print('done')

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print('Usage: python make_wiki_corpus.py <processed_text_file>')
		sys.exit(1)
	out_f = sys.argv[1]
	make_corpus(out_f)

