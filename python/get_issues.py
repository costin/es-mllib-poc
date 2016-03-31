import elasticsearch,urllib2,json, datetime, time


for i in range(0,20000):
    token = HERE BE A TOKEN, see https://help.github.com/articles/creating-an-access-token-for-command-line-use/
    try:
       issue_response = urllib2.urlopen('https://api.github.com/repos/elastic/elasticsearch/issues/' + str(i)+'?access_token='+token)
       comments_response=urllib2.urlopen('https://api.github.com/repos/elastic/elasticsearch/issues/' + str(i)+'/comments?access_token='+token)
       issue = json.loads(issue_response.read())
       comments = json.loads(comments_response.read())
       comments.append(issue)
       text_file = open('/data/github/issues/'+str(i)+".txt", "w")
       text_file.write(json.dumps(comments))
       text_file.close()
       print 'Retrieved issue ' + str(i)
       print str(comments_response.info()['X-RateLimit-Remaining']) + ' requests left before rate limit exceeded'
    except urllib2.URLError:
       print str(i) + ' is not an issue'
        
        
    if int(comments_response.info()['X-RateLimit-Remaining']) < 2:
        # this does not actually worked but before I got to fix it the download was done so I never bothered
        t_now = datetime.datetime.now()
        t_reset = datetime.datetime.utcfromtimestamp(comments_response.info()['X-RateLimit-Remaining'])
        t_diff = t_reset-t_now
        print 'will sleep for ' + str(t_diff.seconds) + 's'
        time.sleep(timedelta.total_seconds())
        while json.loads(urllib2.urlopen('https://api.github.com/rate_limit?access_token='+token).read())['rate']['remaining'] == 0:
            print 'waiting for rate limit to drop to 0'
