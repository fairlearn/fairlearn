from bokeh.plotting import figure, show
import datetime
from dateutil import relativedelta
from dateutil.parser import isoparse
import os

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

base_query_string = '''
{
  repository(owner: "fairlearn", name: "fairlearn") {
    stargazers(first: 50, REPLACE_CURSOR) {
      edges {
        starredAt
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
}
'''


def get_client(token):
    # Note that for an Auth token, we have to have "token" in the header value
    transport = RequestsHTTPTransport(
        url='https://api.github.com/graphql',
        headers={'Authorization': "token {0}".format(token)})
    client = Client(
        transport=transport,
        fetch_schema_from_transport=True)
    return client


def fetch_all_stars(client):
    result = []
    have_more_results = True
    from_cursor = ""
    while have_more_results:
        query_string = base_query_string.replace("REPLACE_CURSOR", from_cursor)
        query = gql(query_string)
        next_results = client.execute(query)

        next_stars = next_results['repository']['stargazers']['edges']
        result.extend([x['starredAt'] for x in next_stars])

        page_info = next_results['repository']['stargazers']['pageInfo']
        if page_info['hasNextPage']:
            have_more_results = True
            from_cursor = 'after: "{0}"'.format(page_info['endCursor'])
        else:
            have_more_results = False

    return result


def process_stars(star_dates):
    by_month = {}

    next_date = isoparse(star_dates[0])
    end_date = datetime.datetime.now(datetime.timezone.utc) + \
        relativedelta.relativedelta(months=1)
    while next_date < end_date:
        stats_dict = {}
        stats_dict['stars'] = 0
        stats_dict['cumulative'] = 0
        month_str = "{0}-{1:02}".format(next_date.year, next_date.month)
        by_month[month_str] = stats_dict
        next_date = next_date + relativedelta.relativedelta(months=1)

    for star_date in star_dates:
        starred = isoparse(star_date)
        month_str = "{0}-{1:02}".format(starred.year, starred.month)
        by_month[month_str]['stars'] += 1

    accumulator = 0
    for v in by_month.values():
        accumulator += v['stars']
        v['cumulative'] = accumulator

    return by_month


def plot_stars(stats):
    month_list = list(stats.keys())
    delta_stars = [x['stars'] for x in stats.values()]
    total_stars = [x['cumulative'] for x in stats.values()]

    p = figure(x_range=month_list,
               plot_height=512,
               plot_width=832,
               title="Stars",
               toolbar_location=None,
               tools="")

    p.vbar(x=month_list, top=delta_stars, width=0.9)
    p.line(x=month_list, y=total_stars, color="red", line_width=2)

    p.y_range.start = 0
    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.xaxis.major_label_orientation = 45.0
    p.outline_line_color = None

    show(p)


def github_star_gazing():
    print("Getting client")
    client = get_client(os.environ["GITHUB_GRAPHQL"])
    print("Getting data from GitHub")
    star_dates = fetch_all_stars(client)
    print("Found ", len(star_dates), " total stars")
    stats = process_stars(star_dates)
    plot_stars(stats)


print("Starting script")
github_star_gazing()
