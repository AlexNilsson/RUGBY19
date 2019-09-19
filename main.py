import os

import numpy as np

from utility import csv_to_dict, keep_entries, group_by_key, unique_members_from_columns
import config as C

from RUGBYNET import RUGBYNET

""" FILE PATHS """
PATH_TO_ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = os.path.join(PATH_TO_ROOT, 'data')
PATH_TO_PREDICT = os.path.join(PATH_TO_ROOT, 'predict')

""" LOAD DATA """
WORLD_CUP_MATCHES = csv_to_dict(os.path.join(PATH_TO_DATA, 'Matches.csv'))

PREDICT_MATCHES = csv_to_dict(os.path.join(PATH_TO_PREDICT, 'matches.csv'))

""" PREPROCESS DATA"""
# EXTRACT FEATURES OF INTEREST
WORLD_CUP_MATCHES = keep_entries(WORLD_CUP_MATCHES, ['Home Team Name','Home Team Goals','Away Team Goals', 'Away Team Name', 'Date'])

# EXTRACT LIST OF ALL TEAMS
# ALL_TEAMS =  unique_members_from_columns(WORLD_CUP_MATCHES, ['Home Team Name', 'Away Team Name'])

# All team names to upper case
for match in WORLD_CUP_MATCHES:
  match['Home Team Name'] = match['Home Team Name'].upper()
  match['Away Team Name'] = match['Away Team Name'].upper()
  if (len(match['Date']) > 1):
    match['Date'] = match['Date'].split(' ')[2] # day month year -> year


# The teams we care about
TEAMS_TO_PREDICT = list(set(
  [item for match in PREDICT_MATCHES for item in list(match.values())]
))


# Filter out relevant matches
WORLD_CUP_MATCHES = list(
  filter(lambda match:
  (match['Home Team Name'] in TEAMS_TO_PREDICT) and
  (match['Away Team Name'] in TEAMS_TO_PREDICT) and
  (len(match['Date']) == 4)
  , WORLD_CUP_MATCHES)
)

# Remove matches with missing scores, or where results are 0-0 (prob. invalid)
WORLD_CUP_MATCHES = list(
  filter(lambda match:
  match['Home Team Goals'].isdigit() and
  match['Away Team Goals'].isdigit() and
  ((int(match['Home Team Goals']) +
  int(match['Away Team Goals'])) > 0)
  , WORLD_CUP_MATCHES)
)


# ONE HOT ENCODINGS OF ALL TEAMS
ALL_TEAMS_ENCODING = dict(zip(TEAMS_TO_PREDICT, np.eye(len(TEAMS_TO_PREDICT))))

# GROUP BY YEAR/CUP
#print (WORLD_CUP_MATCHES)
WORLD_CUP_MATCHES = group_by_key(WORLD_CUP_MATCHES, 'Date')

# Decay scores to account for athletic improvements over time
""" N_GAMES = len(WORLD_CUP_MATCHES)
for i, year in enumerate(WORLD_CUP_MATCHES):
  scoreDecay = 1 - C.SCORE_DECAY_FACTOR * (N_GAMES-i-1)/(N_GAMES-1)
  for i, match in enumerate(WORLD_CUP_MATCHES[year]):
    try:
      match[2] = scoreDecay * int(match[2])
      match[3] = scoreDecay * int(match[2])
    except:
      # Remove matches written in the wrong format in the database
      # Some entries have home team & scores mixed up and can not be trusted :(
      del match """

# Matches to Predict (1 hot encoded)
temp = []
for x in PREDICT_MATCHES:
  teams = list(x.values())
  match = []

  temp.append(list(np.append(
    ALL_TEAMS_ENCODING[teams[0]],
    ALL_TEAMS_ENCODING[teams[1]]
  )))

PREDICT_MATCHES = np.array(temp, dtype=int) # float to int

""" SPLIT DATA INTO TRAIN/TEST SETS """
""" AND SEPARATE FEATURES FROM LABELS """
if C.USE_TEST_DATA:
  TEST_DATA_RAW = []
  for TEST_YEAR in ['2015', '2016', '2017', '2018', '2019']:
    for m in WORLD_CUP_MATCHES.pop(TEST_YEAR):
      TEST_DATA_RAW.append(m)

      
TRAIN_DATA_RAW = []
for x in WORLD_CUP_MATCHES:
  for m in WORLD_CUP_MATCHES[x]:
    TRAIN_DATA_RAW.append(m)


# Raw data on format: [teamA, scoreA, scoreB, teamB]

TRAIN_DATA_FEATURES = []
TRAIN_DATA_LABELS = []
for x in TRAIN_DATA_RAW:
  try:
    combined = np.append( ALL_TEAMS_ENCODING[x[0]], ALL_TEAMS_ENCODING[x[3]] ) # [teamA, teamB] -> combined hot encoded vector, 2 ones.
    TRAIN_DATA_LABELS.append(list(combined))

    TRAIN_DATA_FEATURES.append(
      list(np.array([x[1], x[2]], dtype=float)) # [scoreA, scoreB]
    )

  except: continue

TRAIN_DATA_FEATURES = np.array(TRAIN_DATA_FEATURES, dtype=float)
TRAIN_DATA_LABELS = np.array(TRAIN_DATA_LABELS, dtype=int)

if C.USE_TEST_DATA:
  TEST_DATA_FEATURES = []
  TEST_DATA_LABELS = []
  for x in TEST_DATA_RAW:
    try:
      combined = np.append(ALL_TEAMS_ENCODING[x[0]], ALL_TEAMS_ENCODING[x[3]]) # [teamA, teamB] -> combined hot encoded vector, 2 ones.
      TEST_DATA_LABELS.append(list(combined))

      TEST_DATA_FEATURES.append(
        list(np.array([x[1], x[2]], dtype=float)) # [scoreA, scoreB]
      )
    except: continue

  TEST_DATA_FEATURES = np.array(TEST_DATA_FEATURES, dtype=float)
  TEST_DATA_LABELS = np.array(TEST_DATA_LABELS, dtype=int)

""" EXECUTE """
rugbyNet = RUGBYNET()

if C.USE_TEST_DATA:
  rugbyNet.train(
    np.array(TRAIN_DATA_LABELS),
    np.array(TRAIN_DATA_FEATURES),
    np.array(TEST_DATA_LABELS),
    np.array(TEST_DATA_FEATURES),
    epochs = C.EPOCHS,
    batch_size = C.BATCH_SIZE
  )

else:
  rugbyNet.train(
    np.array(TRAIN_DATA_LABELS),
    np.array(TRAIN_DATA_FEATURES),
    np.array(TRAIN_DATA_LABELS),
    np.array(TRAIN_DATA_FEATURES),
    epochs = C.EPOCHS,
    batch_size = C.BATCH_SIZE
  )

rugbyNet.predict(np.array(PREDICT_MATCHES))
