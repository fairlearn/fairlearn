# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.


from bokeh.plotting import figure, show
import datetime
from dateutil import relativedelta
from dateutil.parser import isoparse
import os

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

_SECRET_ENV_VAR = "GITHUB_GRAPHQL"


base_issue_query = '''
{
  repository(owner: "fairlearn", name: "fairlearn") {
    issues(first: 50, REPLACE_CURSOR) {
      edges {
        node {
          createdAt
          author {
            login
          }
        }
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}

'''

_CREATE_KEY = 'createdAt'
_AUTHOR_KEY = 'author'

_ALL_KEY = 'all'
_FILTERED_KEY = 'filtered'

_TROUBLEMAKERS = ['MiroDudik', 'romanlutz', 'riedgar-ms']

def get_client(token):
    # Note that for an Auth token, we have to have "token" in the header value
    transport = RequestsHTTPTransport(
        url='https://api.github.com/graphql',
        headers={'Authorization': "token {0}".format(token)})
    client = Client(
        transport=transport,
        fetch_schema_from_transport=True)
    return client


def fetch_all_issues(client):
    result = []
    have_more_results = True
    from_cursor = ""
    
    while have_more_results:
        query_string = base_issue_query.replace("REPLACE_CURSOR", from_cursor)
        query = gql(query_string)
        next_results = client.execute(query)

        raw_issues = next_results['repository']['issues']['edges']
        for issue in raw_issues:
            processed_issue = dict()
            processed_issue[_CREATE_KEY] = issue['node']['createdAt']
            processed_issue[_AUTHOR_KEY] = issue['node']['author']['login']
            result.append(processed_issue)

        page_info = next_results['repository']['issues']['pageInfo']
        if page_info['hasNextPage']:
            have_more_results = True
            from_cursor = 'after: "{0}"'.format(page_info['endCursor'])
        else:
            have_more_results = False

    return result

def _get_month_string(target_date):
    return "{0}-{1:02}".format(target_date.year, target_date.month)

def process_issues(issues):
    by_month = {}

    # The differences in replaced day are to ensure that the bins include
    # the current month, while respecting the times of day of each date object
    next_date = isoparse(issues[0][_CREATE_KEY]).replace(day=1)
    end_date = datetime.datetime.now(datetime.timezone.utc).replace(day=2)
    while next_date < end_date:
        stats_dict = {}
        stats_dict[_ALL_KEY] = 0
        stats_dict[_FILTERED_KEY] = 0
        by_month[_get_month_string(next_date)] = stats_dict
        next_date = next_date + relativedelta.relativedelta(months=1)

    for issue in issues:
        month_str = _get_month_string(isoparse(issue[_CREATE_KEY]))
        by_month[month_str][_ALL_KEY] += 1
        if issue[_AUTHOR_KEY] not in _TROUBLEMAKERS:
            by_month[month_str][_FILTERED_KEY] += 1
        else:
            print(issue[_AUTHOR_KEY])

    return  by_month



def github_issue_creation():
    print("Getting client")
    client = get_client(os.environ[_SECRET_ENV_VAR])
    print("Getting data from GitHub")
    issues = fetch_all_issues(client)
    print("Found {0} issues in total".format(len(issues)))
    issues_by_month = process_issues(issues)



def create_no_fetch_data_plot():
    xs = ["Unable", "to", "fetch", "data"]
    ys = range(len(xs))
    p = figure(x_range=xs,
               plot_height=512,
               plot_width=832,
               title="Stars",
               toolbar_location=None,
               tools="")
    p.vbar(x=xs, top=ys, width=0.5)

    show(p)


print("Starting script {0}".format(__file__))
if _SECRET_ENV_VAR in os.environ:
    github_issue_creation()
else:
    print("Did not find environment variable {0}".format(_SECRET_ENV_VAR))
    create_no_fetch_data_plot()
print("Script {0} complete".format(__file__))
