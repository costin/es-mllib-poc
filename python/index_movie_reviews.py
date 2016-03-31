import json, requests, pprint, random, math, operator, datetime, sys, optparse, time, elasticsearch, os
def create_index():
    es = elasticsearch.Elasticsearch()
    es.indices.delete("movie-reviews")
    mapping = {
       "mappings": {
          "review": {
             "properties": {
                "text": {
                    "type": "string",
                    "term_vector": "with_positions_offsets_payloads"
                }
             }
          }
       },
       "settings": {
          "index.number_of_shards": 1
       }
    }
    es.indices.create(index="movie-reviews",body=mapping)



# download data from: http://www.cs.cornell.edu/people/pabo/movie-review-data/ 
# Need to change the path to the data below



def gather_filenames():
    allPosFiles = []
    print os.getcwd()
    for r,d,f in os.walk('./txt_sentoken/pos'):
        for files in f:
           allPosFiles.append(os.path.join(r,files))

    allNegFiles = []
    for r,d,f in os.walk('./txt_sentoken/neg'):
        for files in f:
           allNegFiles.append(os.path.join(r,files))
    return [allPosFiles, allNegFiles]

# create a bulk request from the filenames
def indexDocsBulk(filenames, classlabels, docIdStart):
    bulk_string = ''
    random.seed()
    docId = docIdStart;
    es = elasticsearch.Elasticsearch()
    for filename in filenames :
        f = open(filename, 'r')
        #header for bulk request
        header = "{ \"index\" : { \"_index\" : \"movie-reviews\", \"_type\" : \"review\", \"_id\": \""+str(docId)+"\"} }"
        # text process: remove all newlines and secial characters
        text = f.read().replace('\n', ' ').replace('"', ' ').replace("\\"," ")
        text = "".join([i for i in text if 31 < ord(i) < 127])
        #create the document text
        doc = "{\"text\": \"" + text + "\",\"class\": \""+ classlabels + "\"}"
        #add to the bulk request
        bulk_string += (header + "\n")
        bulk_string += (doc + "\n")  
        docId += 1;
    response = es.bulk(body=bulk_string, refresh=True)
    print "Bulk took " + str(float(response['took'])/1000.0) + " s"
    return docId



# index data
def index_docs(allPosFiles, allNegFiles):
    nextId = indexDocsBulk(allPosFiles, "pos", 1)
    indexDocsBulk(allNegFiles, "neg", nextId)
    
if __name__ == "__main__":
    create_index()
    [pos_files, neg_files] = gather_filenames()
    index_docs(pos_files, neg_files)
