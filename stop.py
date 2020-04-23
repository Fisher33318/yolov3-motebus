import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--stream_name", type=str, help="stream name")
data_json = {"stream_name": '%s' %parser.parse_args().stream_name, "method": 'stop'}
with open('cmd.json', 'w') as f:
    json.dump(data_json, f)
