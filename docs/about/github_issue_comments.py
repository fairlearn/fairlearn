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


def fetch_all_issue_comments(client):
    result = []

    issue_numbers = fetch_issue_numbers(client)
    print("Found {0} issues".format(len(issue_numbers)))

    return result


def github_issue_comments():
    print("Getting client")
    client = get_client(os.environ[SECRET_ENV_VAR])
    print("Getting data from GitHub")
    issues = fetch_all_issue_comments(client)
    print("Found {0} issues in total".format(len(issues)))
    # issues_by_month = process_issue_comments(issues)
    print("Issues processed")
    # plot_issue_comments(issues_by_month)
    print("Done")


print("Starting script {0}".format(__file__))
if SECRET_ENV_VAR in os.environ:
    github_issue_comments()
else:
    print("Did not find environment variable {0}".format(SECRET_ENV_VAR))
    create_no_fetch_data_plot()
print("Script {0} complete".format(__file__))
