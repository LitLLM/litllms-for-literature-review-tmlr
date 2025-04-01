# https://github.com/allenai/s2-folks/blob/main/examples/python/s2ag_datasets/
# Shows how to download and inspect data in the sample datasets
# which are much smaller than the full datasets.
import json
import subprocess
import gzip

# subprocess.check_call("bash get_sample_files.sh", shell=True)
savedir="/results/auto_review/dataset/"
# download_path = f"{savedir}/papers-part{index}.jsonl"

papers = [json.loads(l) for l in open(f"{savedir}/samples/papers/papers-sample.jsonl", "r").readlines()]
citations = [json.loads(l) for l in open(f"{savedir}/samples/citations/citations-sample.jsonl", "r").readlines()]
# embeddings = [json.loads(l) for l in open("samples/embeddings/embeddings-sample.jsonl", "r").readlines()]

# S2ORC -- only 8 papers
docs = [json.loads(l) for l in open(f"{savedir}/samples/s2orc/s2orc-sample.jsonl", "r").readlines()]
index = 2
text = docs[index]['content']['text']
annotations = {k: json.loads(v) for k, v in docs[index]['content']['annotations'].items() if v}

# print(f"Text: {text} \n")
# print(annotations)
# for a in annotations['paragraph'][:10]:
#     print(a)
# for a in annotations['bibref'][:10]:
#     print(a)
for a in annotations['bibentry'][:10]:
    print(a)
for a in annotations['bibauthor'][:10]:
    print(a)
for a in annotations['bibtitle'][:10]:
    print(a)
for a in annotations['bibref'][:10]:
    print(a)



def text_of(type):
    return [text[a['start']:a['end']] for a in annotations.get(type, '')]

# print(text_of('abstract'))
# print('\n\n'.join(text_of('paragraph')[:3]))
# print('\n'.join(text_of('bibref')[:10]))
# print('\n'.join(text_of('bibentry')[:10]))

# SA: TODO This works
# For the S2ORC dataset:
print("\nPaper keys: ", docs[index].keys())
# Paper keys:  dict_keys(['corpusid', 'externalids', 'content'])
print("\nContent keys: ", docs[index]['content'].keys())
print("\nSource: ", docs[index]['content']['source'])
# Source:  {'pdfurls': ['https://export.arxiv.org/pdf/2306.14666v3.pdf'], 'pdfsha': 'e36a24dde78955a4e6470554df77e497d91097b8', 'oainfo': None}
print("\n Annotation keys: ", annotations.keys())
#  Annotation keys:  dict_keys(['abstract', 'author', 'authoraffiliation', 'authorfirstname', 'authorlastname', 
# 'bibauthor', 'bibauthorfirstname', 'bibauthorlastname', 'bibentry', 'bibref', 'bibtitle', 'bibvenue', 'figure', 
# 'figurecaption', 'figureref', 'formula', 'paragraph', 'sectionheader', 'table', 'title'])
print("\n Annotation sectionheader: ", annotations["sectionheader"])
print("\n Annotation title: ", annotations["title"])
print("\n Annotation title text: ", text_of("title"))
# print("\nText: ", docs[0]['content']['text'])
print("\n Paragraph: ", annotations["paragraph"])
print("\n Sections:", text_of('sectionheader'))
print("\n corpusid: ", docs[index]['corpusid'])
print("\n externalids: ", docs[index]['externalids'])
# externalids:  {'arxiv': '2306.14666', 'mag': None, 'acl': None, 'pubmed': None, 'pubmedcentral': None, 'dblp': None, 'doi': None}

print("\n Annotation bibref: ", annotations["bibref"])
print("\n Annotation bibentry: ", annotations["bibentry"])

print("\n Annotation bibref some: ", annotations["bibref"][:5])
print("\n Annotation bibentry some: ", annotations["bibentry"][:5])

print("\n Annotation bibref first: ", annotations["bibref"][0]["start"])
print("\n Annotation bibref some: ", annotations["bibref"][0]["attributes"]["ref_id"])

print("\n Annotation bibentry first: ", annotations["bibentry"][0]["start"])
print("\n Annotation bibentry some: ", annotations["bibentry"][0]["attributes"]["matched_paper_id"])
# you get sectionheader annotations -- 

def explore_papers(papers):
    # SA: TODO This works
    # For papers
    print("\nPaper keys: ", papers[0].keys())
    # Paper keys:  dict_keys(['corpusid', 'externalids', 'url', 'title', 'authors', 'venue', 
    # 'publicationvenueid', 'year', 'referencecount', 'citationcount', 'influentialcitationcount', 
    # 'isopenaccess', 's2fieldsofstudy', 'publicationtypes', 'publicationdate', 'journal'])
    print("\n publicationdate: ", papers[0]["publicationdate"])
    print("\n title: ", papers[0]["title"])
    print("\n corpusid: ", papers[0]["corpusid"])
    print("\n externalids: ", papers[0]["externalids"])
# explore_papers(papers)

def explore_citations(citations):
    # For citations
    key = 1
    print("\n citations keys: ", citations[key].keys())
    print("\n citationid: ", citations[key]["citationid"])
    print("\n citingcorpusid: ", citations[key]["citingcorpusid"])
    print("\n citedcorpusid: ", citations[key]["citedcorpusid"])
    print("\n isinfluential: ", citations[key]["isinfluential"])
    print("\n contexts: ", citations[key]["contexts"])
    print("\n intents: ", citations[key]["intents"])

# explore_citations(citations)


def inspect_papers_dataset(config):
    index = 0
    jsonl_path = f"{config.data_dir}/papers-part{index}.jsonl"
    papers = [json.loads(l) for l in open(jsonl_path, "r").readlines()]
    print("\n Total paper: ", len(papers))
    print("\nPaper keys: ", papers[0].keys())
    # Paper keys:  dict_keys(['corpusid', 'externalids', 'url', 'title', 'authors', 
    # 'venue', 'publicationvenueid', 'year', 'referencecount', 'citationcount', 
    # 'influentialcitationcount', 'isopenaccess', 's2fieldsofstudy', 'publicationtypes', 
    # 'publicationdate', 'journal'])        


def read_jsonl_files(jsonl_file_path):
    with gzip.open(jsonl_file_path, 'rb') as gzipped_file:
        json_lines = gzipped_file.readlines()
        docs = []
        for line in json_lines:
            docs.append(json.loads(line.decode('utf-8')))
            # Process each JSON object in the file
    return docs
