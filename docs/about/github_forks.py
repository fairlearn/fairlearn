# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.


from bokeh.plotting import figure, show
import datetime
from dateutil import relativedelta
from dateutil.parser import isoparse
import os

from gql import gql

from github_stats_utils import create_no_fetch_data_plot, get_client, get_month_string
from github_stats_utils import SECRET_ENV_VAR


base_query_string = '''
{
  repository(owner: "fairlearn", name: "fairlearn") {
    forks(first: 50, REPLACE_CURSOR) {
      edges {
        node {
            createdAt
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

_FORKS_KEY = 'forks'
_CUMULATIVE_KEY = 'cumulative'


def fetch_all_forks(client):
    result = []
    have_more_results = True
    from_cursor = ""
    while have_more_results:
        query_string = base_query_string.replace("REPLACE_CURSOR", from_cursor)
        query = gql(query_string)
        next_results = client.execute(query)

        next_forks = next_results['repository']['forks']['edges']
        result.extend([x['node']['createdAt'] for x in next_forks])

        page_info = next_results['repository']['forks']['pageInfo']
        if page_info['hasNextPage']:
            have_more_results = True
            from_cursor = 'after: "{0}"'.format(page_info['endCursor'])
        else:
            have_more_results = False

    return result


def process_forks(fork_dates):
    by_month = {}

    # The differences in replaced day are to ensure that the bins include
    # the current month, while respecting the times of day of each date object
    next_date = isoparse(list(sorted(fork_dates))[0]).replace(day=1)
    end_date = datetime.datetime.now(datetime.timezone.utc).replace(day=2)
    while next_date < end_date:
        stats_dict = {}
        stats_dict[_FORKS_KEY] = 0
        stats_dict[_CUMULATIVE_KEY] = 0
        month_str = get_month_string(next_date)
        by_month[month_str] = stats_dict
        next_date = next_date + relativedelta.relativedelta(months=1)

    for f_d in fork_dates:
        forked = isoparse(f_d)
        month_str = get_month_string(forked)
        by_month[month_str][_FORKS_KEY] += 1

    accumulator = 0
    for v in by_month.values():
        accumulator += v[_FORKS_KEY]
        v[_CUMULATIVE_KEY] = accumulator
    return by_month


def plot_forks(stats):
    month_list = list(stats.keys())
    delta_forks = [x[_FORKS_KEY] for x in stats.values()]
    total_forks = [x[_CUMULATIVE_KEY] for x in stats.values()]

    p = figure(x_range=month_list,
               plot_height=512,
               plot_width=832,
               title="Forks",
               toolbar_location=None,
               tools="")

    p.vbar(x=month_list, top=delta_forks, width=0.9)
    p.line(x=month_list, y=total_forks, color="red", line_width=2)

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.xaxis.major_label_orientation = 45.0
    p.outline_line_color = None

    show(p)


def github_fork_counting():
    print("Getting client")
    client = get_client(os.environ[SECRET_ENV_VAR])
    print("Getting data from GitHub")
    fork_dates = fetch_all_forks(client)
    print("Found ", len(fork_dates), " total forks")
    stats = process_forks(fork_dates)
    print("Creating plot")
    plot_forks(stats)


print("Starting script {0}".format(__file__))
if SECRET_ENV_VAR in os.environ:
    github_fork_counting()
else:
    print("Did not find environment variable {0}".format(SECRET_ENV_VAR))
    create_no_fetch_data_plot()
print("Script {0} complete".format(__file__))
