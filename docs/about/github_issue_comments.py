# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.


from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.transform import dodge
import datetime
from dateutil import relativedelta
from dateutil.parser import isoparse
from gql import gql
import os

from github_stats_utils import create_no_fetch_data_plot, get_client, get_month_string
from github_stats_utils import SECRET_ENV_VAR

base_issue_number_query = '''
{
  repository(owner: "fairlearn", name: "fairlearn") {
    issues(first: 50, REPLACE_CURSOR) {
      edges {
        node {
          number
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

base_issue_comment_query = '''
{
  repository(owner: "fairlearn", name: "fairlearn") {
    issue(number: ISSUE_NUMBER) {
      createdAt
      comments(first: 50, REPLACE_CURSOR) {
        nodes{
          createdAt,
          author {
            login
          }
        }
        pageInfo {
          endCursor,
          hasNextPage
        }
      }
    }
  }
}
'''

_CREATE_KEY = "createdAt"
_AUTHOR_KEY = "Author"

_ALL_KEY = 'all'
_FILTERED_KEY = 'filtered'

_TROUBLEMAKERS = ['MiroDudik', 'romanlutz', 'riedgar-ms']


def fetch_issue_numbers(client):
    result = []
    have_more_results = True
    from_cursor = ""

    while have_more_results:
        query_string = base_issue_number_query.replace("REPLACE_CURSOR", from_cursor)
        query = gql(query_string)
        next_results = client.execute(query)

        raw_issues = next_results['repository']['issues']['edges']
        for issue in raw_issues:
            result.append(issue['node']['number'])

        page_info = next_results['repository']['issues']['pageInfo']
        if page_info['hasNextPage']:
            have_more_results = True
            from_cursor = 'after: "{0}"'.format(page_info['endCursor'])
        else:
            have_more_results = False

    return result


def fetch_comments_for_issue(client, issue_number):
    result = []
    have_more_results = True
    from_cursor = ""
    issue_comment_query_string = base_issue_comment_query.replace('ISSUE_NUMBER',
                                                                  str(issue_number))

    while have_more_results:
        query_string = issue_comment_query_string.replace("REPLACE_CURSOR", from_cursor)
        query = gql(query_string)
        next_results = client.execute(query)

        raw_comments = next_results['repository']['issue']['comments']['nodes']
        for rc in raw_comments:
            processed_comment = dict()
            processed_comment[_CREATE_KEY] = rc['createdAt']
            processed_comment[_AUTHOR_KEY] = rc['author']['login']
            result.append(processed_comment)

        page_info = next_results['repository']['issue']['comments']['pageInfo']
        if page_info['hasNextPage']:
            have_more_results = True
            from_cursor = 'after: "{0}"'.format(page_info['endCursor'])
        else:
            have_more_results = False

    return result


def fetch_all_issue_comments(client):
    result = []

    issue_numbers = fetch_issue_numbers(client)
    print("Found {0} issues".format(len(issue_numbers)))

    for i in issue_numbers:
        comments = fetch_comments_for_issue(client, i)
        result.extend(comments)

    return result


def process_issue_comments(issues):
    by_month = {}

    # The differences in replaced day are to ensure that the bins include
    # the current month, while respecting the times of day of each date object
    next_date = isoparse(issues[0][_CREATE_KEY]).replace(day=1)
    end_date = datetime.datetime.now(datetime.timezone.utc).replace(day=2)
    while next_date < end_date:
        stats_dict = {}
        stats_dict[_ALL_KEY] = 0
        stats_dict[_FILTERED_KEY] = 0
        by_month[get_month_string(next_date)] = stats_dict
        next_date = next_date + relativedelta.relativedelta(months=1)

    for issue in issues:
        month_str = get_month_string(isoparse(issue[_CREATE_KEY]))
        by_month[month_str][_ALL_KEY] += 1
        if issue[_AUTHOR_KEY] not in _TROUBLEMAKERS:
            by_month[month_str][_FILTERED_KEY] += 1

    return by_month


def plot_issue_comments(stats):
    month_list = list(stats.keys())
    all_comments = [x[_ALL_KEY] for x in stats.values()]
    filtered_comments = [x[_FILTERED_KEY] for x in stats.values()]

    data = {
        'months': month_list,
        _ALL_KEY: all_comments,
        _FILTERED_KEY: filtered_comments
    }
    source = ColumnDataSource(data=data)

    p = figure(x_range=month_list,
               plot_height=512,
               plot_width=832,
               title="Issue Comments Made",
               toolbar_location=None,
               tools="")

    p.vbar(x=dodge('months', -0.2, range=p.x_range),
           top=_ALL_KEY, width=0.3, source=source, color="#c9d9d3", legend_label="All")
    p.vbar(x=dodge('months', 0.2, range=p.x_range),
           top=_FILTERED_KEY, width=0.3, source=source, color="#718dbf", legend_label="Filtered")

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.xaxis.major_label_orientation = 45.0
    p.outline_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    show(p)


def github_issue_comments():
    print("Getting client")
    client = get_client(os.environ[SECRET_ENV_VAR])
    print("Getting data from GitHub")
    issue_comments = fetch_all_issue_comments(client)
    print("Found {0} comments in total".format(len(issue_comments)))
    comments_by_month = process_issue_comments(issue_comments)
    print("Issues processed")
    plot_issue_comments(comments_by_month)
    print("Done")


print("Starting script {0}".format(__file__))
if SECRET_ENV_VAR in os.environ:
    github_issue_comments()
else:
    print("Did not find environment variable {0}".format(SECRET_ENV_VAR))
    create_no_fetch_data_plot()
print("Script {0} complete".format(__file__))
