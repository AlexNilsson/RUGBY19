import csv
from itertools import groupby

import random
from collections import Counter
import heapq


def calc_loss():
  prediction = {
    "group_games": [
      {
        "NEW ZEALAND": random.randint(0,32),
        "SOUTH AFRICA": random.randint(0,32)
      },
      {
        "ITALY": random.randint(0,32),
        "NAMIBIA": random.randint(0,32)
      },
      {
        "IRELAND": random.randint(0,32),
        "SCOTLAND": random.randint(0,32)
      },
      {
        "ENGLAND": random.randint(0,32),
        "TONGA": random.randint(0,32)
      },
      {
        "WALES": random.randint(0,32),
        "GEORGIA": random.randint(0,32)
      },
      {
        "RUSSIA": random.randint(0,32),
        "SAMOA": random.randint(0,32)
      },
      {
        "FIJI": random.randint(0,32),
        "URUGUAY": random.randint(0,32)
      },
      {
        "ITALY": random.randint(0,32),
        "CANADA": random.randint(0,32)
      },
      {
        "ENGLAND": random.randint(0,32),
        "USA": random.randint(0,32)
      },
      {
        "ARGENTINA": random.randint(0,32),
        "TONGA": random.randint(0,32)
      },
      {
        "JAPAN": random.randint(0,32),
        "IRELAND": random.randint(0,32)
      },
      {
        "SOUTH AFRICA": random.randint(0,32),
        "NAMIBIA": random.randint(0,32)
      },
      {
        "GEORGIA": random.randint(0,32),
        "URUGUAY": random.randint(0,32)
      },
      {
        "AUSTRALIA": random.randint(0,32),
        "WALES": random.randint(0,32)
      },
      {
        "SCOTLAND": random.randint(0,32),
        "SAMOA": random.randint(0,32)
      },
      {
        "FRANCE": random.randint(0,32),
        "USA": random.randint(0,32)
      },
      {
        "NEW ZEALAND": random.randint(0,32),
        "CANADA": random.randint(0,32)
      },
      {
        "GEORGIA": random.randint(0,32),
        "FIJI": random.randint(0,32)
      },
      {
        "IRELAND": random.randint(0,32),
        "RUSSIA": random.randint(0,32)
      },
      {
        "SOUTH AFRICA": random.randint(0,32),
        "ITALY": random.randint(0,32)
      },
      {
        "AUSTRALIA": random.randint(0,32),
        "URUGUAY": random.randint(0,32)
      },
      {
        "ENGLAND": random.randint(0,32),
        "ARGENTINA": random.randint(0,32)
      },
      {
        "JAPAN": random.randint(0,32),
        "SAMOA": random.randint(0,32)
      },
      {
        "NEW ZEALAND": random.randint(0,32),
        "NAMIBIA": random.randint(0,32)
      },
      {
        "FRANCE": random.randint(0,32),
        "TONGA": random.randint(0,32)
      },
      {
        "SOUTH AFRICA": random.randint(0,32),
        "CANADA": random.randint(0,32)
      },
      {
        "ARGENTINA": random.randint(0,32),
        "USA": random.randint(0,32)
      },
      {
        "SCOTLAND": random.randint(0,32),
        "RUSSIA": random.randint(0,32)
      },
      {
        "WALES": random.randint(0,32),
        "FIJI": random.randint(0,32)
      },
      {
        "AUSTRALIA": random.randint(0,32),
        "GEORGIA": random.randint(0,32)
      },
      {
        "NEW ZEALAND": random.randint(0,32),
        "ITALY": random.randint(0,32)
      },
      {
        "ENGLAND": random.randint(0,32),
        "FRANCE": random.randint(0,32)
      },
      {
        "IRELAND": random.randint(0,32),
        "SAMOA": random.randint(0,32)
      },
      {
        "NAMIBIA": random.randint(0,32),
        "CANADA": random.randint(0,32)
      },
      {
        "USA": random.randint(0,32),
        "TONGA": random.randint(0,32)
      },
      {
        "WALES": random.randint(0,32),
        "URUGUAY": random.randint(0,32)
      },
      {
        "JAPAN": random.randint(0,32),
        "SCOTLAND": random.randint(0,32)
      }
    ],

    "favourite_team": "JAPAN"
  }


  prediction

  total_scores = dict(
    sum(
      [Counter(game) for game in prediction["group_games"]], Counter()
    )
  )

  n_teams = len(total_scores)

  total_scores_tuples = list(total_scores.items())
  total_scores_tuples.sort(key = lambda x: x[1], reverse = True) # highest first
  top_teams = total_scores_tuples[:8]

  return top_teams

res = calc_loss()
print(res)

def csv_to_dict(csv_file):
  data = []
  
  with open(csv_file) as fin:
    reader = csv.reader(fin, skipinitialspace=True, quotechar="'")

    for keys in reader: break

    for row in reader:
      data.append(dict(zip(keys, row)))

  return data


def keep_entries(list_of_dicts, keys_to_keep):
  out = []
  for row in list_of_dicts:
    out.append({
      key: row[key] for key in keys_to_keep
    })
  return out

def group_by_key(list_of_dicts, key):
  out = {}
  for k, values in groupby(list_of_dicts, key=lambda x:x[key]):
    out[k] = [list(x.values())[1:] for x in values]
  return out

def unique_members_from_columns(list_of_dicts, keys):
  out = []
  for row in list_of_dicts:
    for key in keys:
      out.append(row[key])
  return list(set(out))