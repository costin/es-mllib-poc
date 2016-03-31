from os import listdir
from os.path import isfile, join
import elasticsearch,urllib2,json, time, copy

es = elasticsearch.Elasticsearch()
try:
    es.indices.delete(index='github')
except:
    print 'index github did not exist?'
    
try:
    es.indices.delete(index='github-single-comments')
except:
    print 'index github-single-comments did not exist?'
    
    
mappings = {
  "mappings": {
    "elasticsearch": {
      "properties": {
        "comments": {
          "type": "nested"
        },
        "original_request": {
          "properties": {
            "labels": {
              "properties": {
                "name": {
                  "type": "string",
                  "analyzer": "keyword"
                }
              }
            }
          }
        }
      }
    }
  }
}

mappings_single_comment = {
  "mappings": {
    "elasticsearch": {
      "properties": {
        "labels": {
          "properties": {
            "name": {
              "type": "string",
              "analyzer": "keyword"
            }
          }
        }
      }
    }
  }
}
es.indices.create(index='github', ignore=400, body = mappings)
es.indices.create(index='github-single-comments', ignore=400, body = mappings_single_comment)
path = '/data/github/issues'
issue_files = [f for f in listdir(path) if isfile(join(path, f))]


es =elasticsearch.Elasticsearch()
for issue_file in issue_files:
    f = open(path+'/'+issue_file, 'r')
    try :
        issue = json.loads(f.read())
    except: 
        f = open(path+'/'+issue_file, 'r')
        print f.read()
        raise
    open_comment = copy.deepcopy(issue[len(issue)-1])
    for comment in issue:
        comment['text']=copy.deepcopy(comment['body'])
        if comment.has_key('labels') == False: 
            if open_comment.has_key('labels') == True: 
                comment['labels'] = copy.deepcopy(open_comment['labels'])
        es.index(index="github-single-comments", doc_type="elasticsearch", body=json.dumps(comment))
        
    doc = {}
    doc['original_request']=copy.deepcopy(open_comment)
    doc['comments']=issue
    es.index(index="github", doc_type="elasticsearch", id=int(issue_file.split('.')[0]), body=json.dumps(doc))
    
        
