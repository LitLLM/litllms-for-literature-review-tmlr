from scholarly import scholarly

def get_author():
    # Retrieve the author's data, fill-in, and print
    # Get an iterator for the author results
    search_query = scholarly.search_author('Shubham Agarwal')
    # Retrieve the first result from the iterator
    first_author_result = next(search_query)
    scholarly.pprint(first_author_result)


# from scholarly import ProxyGenerator

# # Set up a ProxyGenerator object to use free proxies
# # This needs to be done only once per session
# pg = ProxyGenerator()
# pg.FreeProxies()
# scholarly.use_proxy(pg)

# Now search Google Scholar from behind a proxy
search_query = scholarly.search_pubs('Perception of physical stability and center of mass of 3D objects')
# print(search_query)
scholarly.pprint(next(search_query))
# # print(len(search_query))


from scholarly import scholarly, ProxyGenerator

pg = ProxyGenerator()
success = pg.FreeProxies()
scholarly.use_proxy(pg)

author = next(scholarly.search_author('Steven A Cholewiak'))
scholarly.pprint(author)
